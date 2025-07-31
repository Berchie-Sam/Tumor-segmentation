# Brain Tumor Segmentation Web App

This repository contains a Gradio-based web interface for brain tumor segmentation and classification using a U-Net + Transformer model.

## 📂 Project Structure
```
Tumor-segmentation/
├── app.py # Gradio app entrypoint
├── src/
│ ├── model.py # Architecture definition
│ ├── preprocess.py # Image loading & transforms
│ └── predict.py # Inference logic
├── checkpoints/ # Serialized models (.pth, .onnx)
├── .gitignore
└── README.md
```

## 🚀 Quick Start
1. **Clone & activate env**  
   ```bash
   git clone https://github.com/YourUsername/Tumor-segmentation.git
   cd Tumor-segmentation
   python -m venv tumor-env
   source tumor-env/bin/activate
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**  
   ```bash
   python app.py
   ```
## ⚙️ Configuration
Model formats:

.pth: PyTorch checkpoint

.onnx: ONNX format (CPU only)

Thresholds & logic:

Segmentation mask threshold (default 0.5)

Classifier threshold (default 0.5)

“Inconclusive” label if mask/classifier disagree

These can be tweaked in src/predict.py.

## 📝 License
This project is licensed under the MIT License.