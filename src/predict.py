import os
import torch
import logging
import numpy as np
import onnxruntime
from src.model import U_Net

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def predict(image, model_filename="checkpoints/best_model.onnx"):
    if os.path.isabs(model_filename) or os.path.dirname(model_filename):
        model_path = model_filename
    else:
        model_path = os.path.join("checkpoints", model_filename)
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        ext = os.path.splitext(model_path)[1].lower()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # -------- PyTorch .pth path --------
        if ext == ".pth":
            model = U_Net(in_channels=3, seg_out_channels=1)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device).eval()
            image = image.to(device)
            with torch.no_grad():
                seg_out, cls_out = model(image)
            seg_mask = seg_out.squeeze(0).cpu().numpy()
            cls_pred = torch.sigmoid(cls_out).cpu().item()

        # -------- ONNX path --------
        elif ext == ".onnx":
            session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            image_np = image.detach().cpu().numpy()
            ort_inputs = { session.get_inputs()[0].name: image_np }
            ort_outs = session.run(None, ort_inputs)
            seg_mask = ort_outs[0].squeeze(0)
            cls_pred = float(torch.sigmoid(torch.tensor(ort_outs[1])))

        else:
            raise ValueError(f"Unsupported model format: {ext}")

        maskp = np.max(seg_mask) > 0.5
        maskn = not maskp
        cls1 = (cls_pred==1)
        if   maskp and cls1:
            label = "Tumor present"
        elif not maskp and not cls1:
            label = "Tumor absent"
        elif maskp or cls1:
            label = "Inconclusive: mask/classifier disagree"
        else:
            label = "Tumor absent"
            
        return seg_mask, label

    except FileNotFoundError as fnf:
        logger.error(fnf)
        raise
    except Exception as e:
        logger.exception("Error during model inference")
        raise RuntimeError(f"Inference failed: {e}") from e