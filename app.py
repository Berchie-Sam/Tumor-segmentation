import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from src.preprocess import preprocess
from src.predict import predict
from src.model import U_Net

def overlay_contour(image: np.ndarray,
                    mask_2d: np.ndarray,
                    level: float    = 0.5,
                    color: str      = "red",
                    linewidth: float= 2.0) -> plt.Figure:
    base = image.copy()

    cmap = "gray" if base.ndim == 2 else None

    fig, ax = plt.subplots()
    ax.imshow(base, cmap=cmap)
    ax.contour(
        mask_2d,
        levels=[level],
        colors=[color],
        linewidths=linewidth
    )
    ax.axis("off")
    ax.set_title("Tumor Contour Overlay")
    return fig

def inference(image):
    input_batch = preprocess(image)
    seg_mask, label = predict(input_batch)
    mask_2d = seg_mask.squeeze()              # -> (H, W)
    mask_uint8 = (mask_2d * 255).astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(mask_uint8, cmap="gray")
    ax.set_title("Predicted Tumor Mask")
    ax.axis("off")
    
    contour_fig = overlay_contour(image, mask_2d)
    return fig, mask_uint8, contour_fig, label

def main():
    iface = gr.Interface(
        fn=inference,
        inputs=gr.Image(type="numpy", label="Input MRI Scan"),
        outputs=[
            gr.Plot(label="Segmentation Plot"),
            gr.Image(type="numpy", label="Mask Image"),
            gr.Plot(label="Contour Plot"),
            gr.Textbox(label="Tumor Classification")
        ],
        title="Brain Tumor Segmentation",
        description="Upload a brain MRI image to get the predicted tumor mask and presence/absence label."
    )
    iface.launch()

if __name__ == "__main__":
    main()