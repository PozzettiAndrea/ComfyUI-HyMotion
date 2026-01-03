# ComfyUI-HyMotion

A ComfyUI implementation of **HY-MOTION 1.0**, featuring high-fidelity human motion generation.

## 🚀 Installation

1. Clone this repository into your `ComfyUI/custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Aero-Ex/ComfyUI-HyMotion
   ```
2. Install the required dependencies:
   ```bash
   cd ComfyUI-HyMotion
   pip install -r requirements.txt
   ```

---

# 🛠️ **HY-MOTION 1.0 – COMFYUI MODEL SETUP**

Follow the steps below to download and place the required models correctly.

## **1. CORE MODELS**

### **🔴 HY-MOTION-1.0 (FULL)**
* **[Download latest.ckpt here](https://huggingface.co/SumitMathur8956/Hy-Motion1.0/resolve/main/hymotion/HY-Motion-1.0/latest.ckpt)**
* **Save to:** `ComfyUI/models/hymotion/HY-Motion-1.0/latest.ckpt`

### **🔵 HY-MOTION-1.0-LITE**
* **[Download latest.ckpt here](https://huggingface.co/SumitMathur8956/Hy-Motion1.0/resolve/main/hymotion/HY-Motion-1.0-Lite/latest.ckpt)**
* **Save to:** `ComfyUI/models/hymotion/HY-Motion-1.0-Lite/latest.ckpt`

---

## **2. TEXT ENCODERS**
**All files below must be placed in:** `ComfyUI/models/text_encoders`

### **🟢 VIT-CLIP TEXT ENCODER**
* **[Download safetensors](https://huggingface.co/SumitMathur8956/Hy-Motion1.0/resolve/main/text_encoders/clip-vit-large-patch14.safetensors)**

### **🟡 QWEN3-8B (FP8)**
* **[Download safetensors](https://huggingface.co/SumitMathur8956/Hy-Motion1.0/resolve/main/text_encoders/Qwen3-8B_fp8.safetensors)**

### **🟣 QWEN3-8B (GGUF)**
* **[Browse GGUF files here](https://huggingface.co/SumitMathur8956/Hy-Motion1.0/tree/main/text_encoders/Qwen3-8B-GGUF)**

---

## 📅 Features
- **3D Viewer:** Integrated Three.js based motion viewer.
- **Modular Nodes:** Flexible pipeline for motion generation.
- **Support for Full and Lite Models.**

## 📜 Credits
This project is based on the [HyMotion](https://github.com/hymotion/HyMotion) research. Specialized ComfyUI implementation by Aero-Ex.
