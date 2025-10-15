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

## Program:
### Step1: Import Necessary Libraries
```python
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
```
### Step2: Transform test and train dataset
```python
train_transform = transforms.Compose([
    transforms.RandomRotation(10),      # randomly rotate image +/- 10 degrees
    transforms.RandomHorizontalFlip(),  # randomly flip left-right 50% of images
    transforms.Resize(224),             # resize shortest side to 224 pixels
    transforms.CenterCrop(224),         # crop to 224x224 at center
    transforms.ToTensor(),              # convert image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225]) # ImageNet std
])
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```
### Step3: Get The path of an dataset and load it
```python
root = r"C:\Users\admin\Downloads\Cat-Dog_Pandas-20251015T024917Z-1-001\Cat-Dog_Pandas"
train_data = datasets.ImageFolder(os.path.join(root, 'Train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(root, 'Valid'), transform=test_transform)
class_names = train_data.classes
print("Classes:", class_names)
print(f"Training images: {len(train_data)}")
print(f"Validation images: {len(test_data)}")
train_loader = DataLoader(train_data, batch_size=3, shuffle=True)
test_loader = DataLoader(test_data, batch_size=3, shuffle=True)
```
### Step4: Import the model
```python
ResNet18model = models.resnet18(pretrained=True)
for param in ResNet18model.parameters():
    param.requires_grad = False
num_features = ResNet18model.fc.in_features  # 512 for ResNet18

ResNet18model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 3)   # 3 classes: cat, dog, panda
)
```
### Step5: Train the model
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ResNet18model.fc.parameters(), lr=0.001)
ResNet18model = ResNet18model.to(device)
epochs = 3
max_trn_batch = 800
max_tst_batch = 300

train_losses, test_losses = [], []
train_correct, test_correct = [], []

start_time = time.time()

for i in range(epochs):
    ResNet18model.train()
    trn_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        if b == max_trn_batch:
            break
        X_train, y_train = X_train.to(device), y_train.to(device)

        y_pred = ResNet18model(X_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.max(y_pred, 1)[1]
        trn_corr += (predicted == y_train).sum()

    train_losses.append(loss.item())
    train_correct.append(trn_corr.item())

    # Evaluation
    ResNet18model.eval()
    tst_corr = 0

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            if b == max_tst_batch:
                break
            X_test, y_test = X_test.to(device), y_test.to(device)

            y_val = ResNet18model(X_test)
            predicted = torch.max(y_val, 1)[1]
            tst_corr += (predicted == y_test).sum()

        val_loss = criterion(y_val, y_test)
        test_losses.append(val_loss.item())
        test_correct.append(tst_corr.item())
```
### Step5: Evaluate The model
```python
val_acc = tst_corr.item() / len(test_data)
torch.save(ResNet18model.state_dict(), 'best_resnet18_model.pth')
print(f"Saved model with val_acc = {val_acc*100:.2f}%")

# Print final accuracy
print(f'Test accuracy: {test_correct[-1]*100/len(test_data):.2f}%')
print(f'Training duration: {time.time() - start_time:.2f} seconds')
```
### Step6: Get the image
```python
image_index = 54
img, label = test_data[image_index]

plt.figure(figsize=(4, 4))
plt.imshow(img.permute(1, 2, 0).cpu().numpy().clip(0, 1))
plt.axis('off')
plt.title("Selected Image")
plt.show()
```
### Step7: Predict the data
```python
ResNet18model.eval()
with torch.no_grad():
    img_input = img.unsqueeze(0).to(device)
    output = ResNet18model(img_input)
    pred = output.argmax(dim=1)

predicted_class = class_names[pred.item()]
true_class = class_names[label]
print(f"True class: {true_class}")
print(f"Predicted class: {predicted_class}")
```
### Step8: Test the model
```python
test_load_all = DataLoader(test_data, batch_size=4, shuffle=False)

all_preds = []
all_labels = []

ResNet18model.eval()
with torch.no_grad():
    for X_test, y_test in test_load_all:
        X_test, y_test = X_test.to(device), y_test.to(device)
        y_val = ResNet18model(X_test)
        predicted = torch.argmax(y_val, dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_test.cpu().numpy())

        del X_test, y_test, y_val, predicted
        torch.cuda.empty_cache()
```
### Step9: Confusion Matrix
```python
# Confusion matrix
arr = confusion_matrix(all_labels, all_preds)
df_cm = pd.DataFrame(arr, index=class_names, columns=class_names)

plt.figure(figsize=(9, 6))
sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
plt.xlabel("Prediction")
plt.ylabel("Ground Truth")
plt.title("Confusion Matrix - ResNet18")
plt.show()
```
### Step10: Save the model
```python
# Save final model
torch.save(ResNet18model.state_dict(), "resnet18_catdogpanda.pth")
print("Model saved as resnet18_catdogpanda.pth")
```
<img width="341" height="84" alt="image" src="https://github.com/user-attachments/assets/a82ec0bb-260a-4905-ba9b-d74eb4ab28f7" /><br>
<img width="445" height="451" alt="image" src="https://github.com/user-attachments/assets/c45d55ab-65bb-424c-85c9-2675da2f35d7" /><br>
<img width="210" height="57" alt="image" src="https://github.com/user-attachments/assets/a2964366-5481-4d45-88d4-5b371a693bdc" /><br>
<img width="779" height="565" alt="image" src="https://github.com/user-attachments/assets/35d60a10-9cd7-4d4f-a01f-9540a0ed3167" />

## Results

The model was trained using **ResNet18** (pretrained on ImageNet) for **3 epochs** with a **batch size of 10** on GPU (NVIDIA GeForce MX550, 2GB VRAM).
The dataset contained labeled images of **cats, dogs, and pandas**, structured into training and testing folders.


| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | ~96.33% |
| Test Accuracy       | ~96.33% |
| Test Loss           | 0.21   |
