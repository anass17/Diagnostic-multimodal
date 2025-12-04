This README is in English. For the French version, see [README.md](README.md).

# Unified Intelligent Medical Image Analysis System

Cancerous Blood Cell Classification (PyTorch)  
Brain Tumor Detection (YOLOv8)

## Project Presentation

In a biomedical imaging laboratory, the objective is to automate the analysis of two critical pathologies:

- Brain tumors from MRI/CT scans
- Blood cancers (leukemias) from peripheral blood smears

The project consists of developing a unified medical image analysis solution based on deep learning, combining:

- A **PyTorch** pipeline for blood cell classification
- A **YOLOv8** pipeline for object detection (tumors) in brain images
- A **Streamlit** interface allowing interactive use of both models

## 1. Cancerous Blood Cell Classification (PyTorch)

### Objective

Build a model based on pre-trained `GoogLeNet` to classify different categories of abnormal blood cells.

### Pipeline Steps

#### 1. Image Loading and Verification

- Load the dataset
- Check allowed extensions: `jpeg`, `jpg`, `bmp`, `png`
- Remove invalid files
- Use try/except to handle corrupted images

#### 2. Explore the Classes

- Each class = one folder
- Display the number of images per class using countplot
- Visualize a few images per class

#### 3. Dataset Split

- Split images into:
    - **70%** → Training
    - **15%** → Validation
    - **15%** → Test
- Then count images in each folder

#### 4. Data Augmentation

- On the training dataset:
    - blur
    - noise
    - horizontal/vertical flip
- Objective:
    - balance classes
    - increase data volume

#### 5. PyTorch Transforms

In `ImageFolder`:
- resizing
- conversion to tensors
- normalization

#### 6. DataLoader

Create `DataLoaders` to:
- Load data in batches
- Shuffle data (shuffle=True)

#### 7. Model

- Load pre-trained GoogLeNet
- Replace the fully connected layer with a network adapted to the dataset’s number of classes

#### 8. Hyperparameters

Define:
- learning rate
- loss function (e.g., CrossEntropyLoss)
- optimizer (e.g., Adam, SGD)

#### 9. Model Training

- Complete training loop
- Validation at each epoch
- Save the best model

#### 10. Evaluation

Measure:
- accuracy
- confusion matrix
- generalization ability on the test set

#### 11. Saving

Save:
- the trained model
- parameters
- normalization stats

## 2. Brain Tumor Detection (YOLOv8)

### Objective

Classify and localize tumors in MRI/CT images using `YOLOv8`.

### Pipeline Steps

#### 1. Visualization of Images and Labels

Display a few images per class with their bounding boxes (annotations .txt)

#### 2. Dataset Preparation

Create a clean folder after filtering:
    - For each image, check if a .txt label exists
    - If label exists → copy to images/train, images/valid, images/test
    - Also copy labels to labels/train, etc.
    - If label missing → display warning and ignore the image

#### 3. YOLO Configuration Files

**data.yaml**

Contains:
- paths (train / valid / test)
- number of classes
- class names
- augmentations disabled

**data2.yaml**

Same content but with augmentations enabled

#### 4. Integrity Check

- Ensure every image has a corresponding label
- Delete any image without label
- Delete any label without image

#### 5. Statistics

Count:
- number of images
- number of labels per split

#### 6. YOLOv8 Training

Define:
- image size
- batch size
- epochs
- learning rate
- base model (yolov8n, yolov8s…)

Launch training

#### 7. Evaluation & Testing

Measure:
- precision
- recall
- mAP
- overall generalization performance

#### 8. Model Saving

Export:
- best.pt
- last.pt

## 3. Streamlit Interface — Unified Model

A `Streamlit` interface allows users to:

- upload an image
- run:
    - blood cell classification `(PyTorch)`
    - brain tumor detection `(YOLOv8)`
- display:
    - predicted class
    - image annotated by YOLO
    - probabilities and model information

## Project Structure

```
📁 Diagnostic-multimodal
│
├── pytorch_model/
│   ├── train.py
│   ├── data/
│   ├── saved_model.pth
│   └── utils.py
│
├── yolo_model/
│   ├── images/
│   ├── labels/
│   ├── data.yaml
│   ├── data2.yaml
│   └── runs/
│
├── streamlit_app/
│   └── app.py
│
├── notebooks/
│   ├── classification_cells.ipynb
│   ├── yolo_preparation.ipynb
│   └── evaluation.ipynb
│
├── requirements.txt
└── README.md
```

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/anass17/Diagnostic-multimodal
cd Diagnostic-multimodal
```

2. Install dependencies:
```Bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run main.py
```

4. Open the application in your browser:
Streamlit will automatically open a local window; otherwise go to: `http://localhost:8501/`

## Conclusion

This project delivers a complete deep learning solution for medical image analysis, combining:

- Classification of blood cell images
- Detection of brain tumors
- A Streamlit dashboard for simplified clinical use

It provides a modern, professional pipeline for AI-assisted medical diagnosis automation.

### Streamlit Interface

![Streamlit UI 1](https://github.com/user-attachments/assets/0ee84e5b-44d8-45a8-b7cc-18ea5df7c5d4)
![Streamlit UI 2](https://github.com/user-attachments/assets/3fac894d-f312-4d18-aa71-cf393c4de206)