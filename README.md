# Image-Classification-using-Resnet18

### Developed By: Ashwin Akash M

## Overview

This project builds an **image classification model** to identify whether an image belongs to a **cat, dog, or panda** using **transfer learning** with **ResNet18** in PyTorch.

We leverage a pre-trained model on ImageNet and fine-tune its final layers to adapt it for this 3-class classification task.

## Setup Instructions

### 1. Create and Activate a Python Environment in Anaconda
```
conda create -n torch_env python=3.10
conda activate torch_env
```
### 2. Install Dependencies

```
pip install -r requirements.txt
```

## Dataset

The dataset is organized into separate training, validation, and testing sets to assess the model’s accuracy and generalization performance.
##### Dataset Link: https://drive.google.com/drive/folders/1RULxsjUArZXb7JInU94_KyH67lKYHUKy?usp=drive_link
##### Folder structure after extraction:

```
data/
  train/
    cat/
    dog/
    panda/
  test/
    cat/
    dog/
    panda/
```



## CUDA & GPU Verification

Before training, ensure GPU is available and configured:

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
```

If `True`, your model will automatically train using GPU for faster computation.



## Model Architecture

We use **ResNet18 (pre-trained on ImageNet)** and replace its final layer with:

* Fully Connected (256 neurons, ReLU, Dropout 0.5)
* Output Layer (3 neurons for cat, dog, panda)

Training configuration:

* **Criterion**: CrossEntropyLoss
* **Optimizer**: Adam (lr = 0.001)
* **Epochs**: 3
* **Batch Size**: 3



## Evaluation

The notebook reports:

* Test Loss and Test Accuracy
* Confusion Matrix Visualization
* Sample Image Predictions

Best model checkpoint is automatically saved as:

```
best_resnet18.pth
```
## Results

The model was trained using **ResNet18** (pretrained on ImageNet) for **3 epochs** with a **batch size of 10** on GPU (NVIDIA GeForce MX550, 2GB VRAM).
The dataset contained labeled images of **cats, dogs, and pandas**, structured into training and testing folders.


| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | ~96.33% |
| Test Accuracy       | ~96.33% |
| Test Loss           | 0.21   |
