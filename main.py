import os, base64, logging, torch
from io import BytesIO
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crack_final")

# Path configurations
APP_DIR = "/home/ssm-user/crack_demo"
WEIGHTS_PATH = f"{APP_DIR}/app/best.pt"

# FastAPI app initialization
app = FastAPI(root_path="/segmentation-engine")
templates = Jinja2Templates(directory=f"{APP_DIR}/templates")

# Mount static files if directory exists
if os.path.exists(f"{APP_DIR}/static"):
    app.mount("/static", StaticFiles(directory=f"{APP_DIR}/static"), name="static")

# Global model and device placeholders
_model = None
_device = None

def _ensure_model_loaded():
    """Initializes and loads the UNet model weights into memory if not already loaded."""
    global _model, _device
    if _model is not None:
        return
    torch.set_num_threads(1)
    from app.unet_model import UNet
    _device = torch.device("cpu")
    model = UNet(in_channels=3, out_channels=1).to(_device)

    ckpt = torch.load(WEIGHTS_PATH, map_location=_device, mmap=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    new_state = {k.replace("final_conv.", "head."): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)
    model.eval()
    _model = model

def img_to_b64(np_img):
    """Converts a numpy image array to a base64 encoded PNG string."""
    if np_img.dtype == bool or np_img.max() <= 1:
        np_img = (np_img * 255).astype(np.uint8)
    img = Image.fromarray(np_img.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Renders the main upload page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    marble: UploadFile = File(...),
    refmask: UploadFile = File(None),
    refmask_b64: str = Form(None),
    low_thr: float = Form(0.2),
    high_thr: float = Form(0.5)
):
    """
    Main analysis endpoint: performs crack segmentation, 
    calculates metrics, and generates an overlay comparison.
    """
    try:
        _ensure_model_loaded()
        from app.infer import to_tensor, hysteresis, refine_mask
        from app.infer_sliding_cloud import sliding_prob
        from app.crack_metrics import crack_table_from_mask, skeletonize_opencv

        MAX_W, MAX_H = 1200, 1200

        # Read and validate image dimensions
        contents = await marble.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        w, h = img.size

        if w > MAX_W or h > MAX_H:
            msg = f"Unsupported image size: {w}x{h}. Max supported is {MAX_W}x{MAX_H}."
            logger.warning(msg)
            return HTMLResponse(
                content=f"<script>alert('{msg}'); window.location.href='/segmentation-engine/';</script>",
                status_code=413,
            )

        rgb = np.array(img)

        # Handle optional Ground Truth mask from file or base64
        ref_mask_img = None
        if refmask and refmask.size > 0:
            ref_bytes = await refmask.read()
            ref_mask_img = Image.open(BytesIO(ref_bytes)).convert("L")
        elif refmask_b64 and "," in refmask_b64:
            _, encoded = refmask_b64.split(",", 1)
            ref_mask_img = Image.open(BytesIO(base64.b64decode(encoded))).convert("L")
            if np.array(ref_mask_img).max() == 0:
                ref_mask_img = None

        # Model inference using sliding window
        x = to_tensor(rgb).to(_device)
        with torch.no_grad():
            prob = sliding_prob(_model, x[0], patch=448, overlap=224, device=_device).cpu().numpy()

        # Post-processing: Hysteresis thresholding and area refinement
        pred01 = hysteresis(prob, low_thr, high_thr)
        pred01 = refine_mask(pred01, min_area=50).astype(np.uint8)

        # Calculate crack properties (length, orientation, width)
        raw_rows = crack_table_from_mask(pred01)
        formatted_rows = [
            {"length_px": r.get("length", 0),
             "orientation_deg": r.get("orientation", 0),
             "mean_width_px": r.get("avg_width", 0)}
            for r in raw_rows
        ]

        skel = skeletonize_opencv(pred01)
        overlay = rgb.copy()

        cm, metrics = None, {"precision": 0, "recall": 0, "f1": 0}
        cm_pct = None

        # Calculate evaluation metrics if Ground Truth is provided
        if ref_mask_img is not None:
            if ref_mask_img.size != (pred01.shape[1], pred01.shape[0]):
                ref_mask_img = ref_mask_img.resize(
                    (pred01.shape[1], pred01.shape[0]), Image.NEAREST
                )

            gt01 = (np.array(ref_mask_img) > 127).astype(np.uint8)

            tp = (pred01 == 1) & (gt01 == 1)
            fp = (pred01 == 1) & (gt01 == 0)
            fn = (pred01 == 0) & (gt01 == 1)
            tn = (pred01 == 0) & (gt01 == 0)

            # Color coding: Green=TP, Red=FP, Blue=FN
            overlay[tp], overlay[fp], overlay[fn] = [0,255,0], [255,0,0], [0,0,255]

            tp_n, fp_n, fn_n, tn_n = int(tp.sum()), int(fp.sum()), int(fn.sum()), int(tn.sum())

            eps = 1e-9
            prec = tp_n / (tp_n + fp_n + eps)
            rec  = tp_n / (tp_n + fn_n + eps)

            cm = {"tp": tp_n, "fp": fp_n, "fn": fn_n, "tn": tn_n}
            metrics = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float((2 * prec * rec) / (prec + rec + eps))
            }

            pos_total = tp_n + fn_n
            neg_total = tn_n + fp_n

            cm_pct = {
                "tp": 100.0 * tp_n / pos_total if pos_total else 0.0,
                "fn": 100.0 * fn_n / pos_total if pos_total else 0.0,
                "tn": 100.0 * tn_n / neg_total if neg_total else 0.0,
                "fp": 100.0 * fp_n / neg_total if neg_total else 0.0,
            }

        else:
            # Simple red overlay if no GT is provided
            overlay[pred01 > 0] = [255, 0, 0]

        return templates.TemplateResponse("result.html", {
            "request": request,
            "marble_b64": img_to_b64(rgb),
            "pred_b64": img_to_b64(pred01),
            "thin_b64": img_to_b64(skel),
            "overlay_base64": img_to_b64(overlay),
            "crack_rows": formatted_rows,
            "n_components": len(formatted_rows),
            "cm": cm,
            "cm_pct": cm_pct,
            "metrics": metrics
        })

    except Exception as e:
        logger.exception("Pipeline Error")
        return HTMLResponse(content=f"Error: {e}", status_code=500)
