import base64
from io import BytesIO
from pathlib import Path


import cv2
import numpy as np
import torch
import logging
import traceback
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from app.unet_model import UNet
from app.model import build_deeplabv3, forward_logits
from app.infer import pil_to_rgb, to_tensor, hysteresis, refine_mask
from app.crack_metrics import crack_table_from_mask, skeletonize_opencv

APP_DIR = "/home/ssm-user/crack_demo"
WEIGHTS_PATH = f"{APP_DIR}/app/best.pt"

app = FastAPI()
templates = Jinja2Templates(directory=f"{APP_DIR}/templates")
app.mount("/static", StaticFiles(directory=f"{APP_DIR}/static"), name="static")

device = torch.device("cpu")



# ---- debug logger ----
logger = logging.getLogger("crack_demo")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.FileHandler("/tmp/crack_demo_debug.log")
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(_h)
logger.propagate = False
# ----------------------

def img_to_b64(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_data_url_png(data_url: str) -> bytes | None:
    if not data_url:
        return None
    try:
        if data_url.startswith("data:"):
            _, b64 = data_url.split(",", 1)
            return base64.b64decode(b64)
        return base64.b64decode(data_url)
    except Exception:
        return None


# Load model once
ckpt = torch.load(WEIGHTS_PATH, map_location=device)
state = ckpt["model"]
model = build_deeplabv3(num_classes=1).to(device)
# SWITCH_TO_UNET_IF_NEEDED
def _is_unet_state(sd: dict) -> bool:
    try:
        ks = list(sd.keys())
        return any(k.startswith("downs.") for k in ks) and any(k.startswith("ups.") for k in ks)
    except Exception:
        return False

try:
    if isinstance(state, dict) and _is_unet_state(state):
        print("Detected UNet checkpoint -> building UNet model")
        model = UNet(in_channels=3, out_channels=1).to(device)
except Exception as e:
    print("UNet auto-detect failed:", e)

model.load_state_dict(state, strict=True)
model.eval()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "hyst_low": 0.2, "hyst_high": 0.50},
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    marble: UploadFile = File(...),
    refmask: UploadFile | None = File(None),
    refmask_b64: str | None = Form(None),
    use_refine: bool = Form(False),
    hyst_low: float = Form(0.2),
    hyst_high: float = Form(0.50),
):
    logger.info("=== /analyze called ===")
    logger.info(f"params: hyst_low={hyst_low} hyst_high={hyst_high} use_refine={use_refine}")
    try:
        logger.info(f"marble filename={getattr(marble,"filename",None)}")
    except Exception:
        pass

    logger.info("=== /analyze called ===")
    logger.info(f"params: hyst_low={hyst_low} hyst_high={hyst_high} use_refine={use_refine}")
    try:
        logger.info(f"marble filename={getattr(marble,"filename",None)}")
    except Exception:
        pass

    # --- read & save marble upload (for eval scripts) ---
    marble_bytes = await marble.read()
    Path(f"{APP_DIR}/data/images").mkdir(parents=True, exist_ok=True)
    safe_name = (marble.filename or "upload.png").replace("/", "_").replace("\\", "_")
    with open(f"{APP_DIR}/data/images/{safe_name}", "wb") as f:
        f.write(marble_bytes)
    # ----------------------------------------------------

    marble_pil = Image.open(BytesIO(marble_bytes))


    marble_rgb = marble_pil.convert("RGB")
    marble_b64 = img_to_b64(marble_rgb)
    rgb = pil_to_rgb(marble_pil)
    x = to_tensor(rgb).to(device)

    with torch.no_grad():
        logits = forward_logits(model, x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    pred = hysteresis(prob, hyst_low, hyst_high).astype(np.uint8)

    if use_refine:
        pred = refine_mask(pred).astype(np.uint8)

    # crack table (length / angle / width)
    crack_rows = crack_table_from_mask(pred)
    thin = skeletonize_opencv(pred)
    thin_img = Image.fromarray((thin * 255).astype(np.uint8))
    thin_b64 = img_to_b64(thin_img)

    overlay = np.array(marble_rgb)
    overlay = cv2.resize(overlay, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)

    # optional reference mask: drawn (b64) has priority over uploaded file
    ref_bytes = None
    if refmask_b64:
        ref_bytes = decode_data_url_png(refmask_b64)

    if ref_bytes is None and refmask is not None and getattr(refmask, "filename", ""):
        b = await refmask.read()
        if b and len(b) > 0:
            ref_bytes = b

    tp = fp = fn = None
    if ref_bytes is not None:
        try:
            ref_pil = Image.open(BytesIO(ref_bytes)).convert("L")
            ref = (np.array(ref_pil) > 127).astype(np.uint8)
            ref = cv2.resize(ref, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

            tp_mask = (pred == 1) & (ref == 1)
            fp_mask = (pred == 1) & (ref == 0)
            fn_mask = (pred == 0) & (ref == 1)

            tp = int(tp_mask.sum())
            fp = int(fp_mask.sum())
            fn = int(fn_mask.sum())

            overlay[tp_mask] = [0, 255, 0]
            overlay[fp_mask] = [255, 0, 0]
            overlay[fn_mask] = [0, 0, 255]
        except Exception:
            overlay[pred == 1] = [255, 0, 0]
    else:
        overlay[pred == 1] = [255, 0, 0]

    pred_img = Image.fromarray((pred * 255).astype(np.uint8))
    overlay_img = Image.fromarray(overlay)

    n_comp, _ = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    n_comp = max(0, n_comp - 1)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "marble_b64": marble_b64,
            "pred_b64": img_to_b64(pred_img),
            "overlay_b64": img_to_b64(overlay_img),
            "thin_b64": thin_b64,
            "n_components": n_comp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "hyst_low": hyst_low,
            "hyst_high": hyst_high,
            "use_refine": use_refine,
            "crack_rows": crack_rows,
        },
    )
