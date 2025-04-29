## Instructions to Run the Project

### Step 0: Download the Dataset From Kaggle

- `link` : https://www.kaggle.com/datasets/felicepollano/watermarked-not-watermarked-images 
- Store the dataset in '\dataset' folder

### Step 1: Install Dependencies
Before running the project, ensure that the required Python packages are installed in your environment. These include:

- torch
- torchvision
- timm
- opencv-python
- matplotlib
- numpy
- pillow

You can install these packages using any Python package manager like pip or conda or simply give the following command:

`pip install -r requirements.txt`

---

### Step 2: Train the Model

1. Open the file `model.ipynb` located in the `notebooks` folder.
2. This notebook trains the ConvNeXt-based classification model using the dataset stored in `dataset/wm-nowm-final/train`.
3. The trained model will be saved as `logoconvnext_best_model.pth` in the same folder.

---

### Step 3: Perform Watermark Detection
1. Open and run the script `model.py` located in the `notebooks` folder.
2. This script will:
   - Load the pre-trained model (`logoconvnext_best_model.pth`).
   - Perform predictions on test images located in `dataset/wm-nowm-final/test`.
   - Output the predicted class (watermark or no-watermark) for each test image.

Make sure the trained model file exists before running the script.

---

## Project Folder Structure

- `dataset/wm-nowm`: Raw dataset
- `dataset/wm-nowm-cleaned/train`: Cleaned and filtered training images
- `dataset/wm-nowm-final/train`: Final training set with watermarked and clean images
- `dataset/wm-nowm-final/val`: Validation images
- `dataset/wm-nowm-final/test`: Test images for evaluation
- `dataset/wm-nowm-final/logos`: Logo images used for synthetic watermark generation
- `notebooks/synthetic_logowm.ipynb`: (Optional) Used for generating synthetic watermark images using logos
- `notebooks/model.ipynb`: Training notebook
- `notebooks/model.py`: Script for performing inference
- `notebooks/logoconvnext_best_model.pth`: Saved trained model file
- `notebooks/evaluate.ipynb`: Evaluated On Test and Validation dataset
- `/Evaluated_Results` : Saved Evaluation Results of the Model

## For Instant Demo of MY MODEL 👇:

### Simple Frontend Integration

- `notebooks/model.py` : Built a simple Streamlit Application (Run : `streamlit run app.py`)



##  Uniqueness of MY Project:

Unlike traditional watermark detection systems that depend solely on manually labeled datasets, this project introduces a **synthetic data generation pipeline** to improve model training diversity and performance.

- A custom notebook (`synthetic_logowm.ipynb`) overlays transparent logos onto clean images to create realistic watermarked samples.
- This enables **cost-effective and scalable augmented data generation**, simulating real world watermarking scenarios.
- It significantly **reduces dependency on scarce annotated datasets** and improves the model's generalization to unseen watermarks.

##  Model Choice: ConvNeXt for Classification

We utilize the **ConvNeXt Tiny model**, a modern and efficient vision transformer-inspired architecture, known for its high performance on image classification tasks.

- Fine-tuned using the synthetically enriched dataset.
- Achieves reliable accuracy in distinguishing **watermarked** vs **non-watermarked** images.
- Saved model weights: `logoconvnext_best_model.pth`

This combination of synthetic data generation and advanced ConvNeXt modeling makes the system both **robust and production-ready**.
