# Kidney Disease Detection using Pre-trained CNN Models

## Overview
This project focuses on detecting kidney diseases from CT scan images using pre-trained Convolutional Neural Networks (CNN) such as VGG16 and ResNet50. By leveraging deep learning and transfer learning, the model extracts features from medical images and classifies them into different kidney conditions: Normal, Cyst, Tumor, and Stone.

## Dataset
The dataset used in this project is the **CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone** from Kaggle. It contains 12,546 CT scan images categorized as:
- **Cyst:** 3,709 images
- **Tumor:** 2,283 images
- **Stone:** 1,377 images
- **Normal:** 5,077 images

## Methodology
1. **Data Preprocessing**
   - Reshaping all images to (224, 224, 3) to match model input requirements.
   - Normalizing pixel values to the range [0, 1] by dividing by 255.
   - Auto-tuning to optimize CPU/GPU performance.

2. **Feature Extraction using Pre-trained Models**
   - **VGG16:** A CNN model with 16 layers, widely used for image classification.
   - **ResNet50:** A deep residual network with 50 layers, designed to address the vanishing gradient problem.

3. **Training and Evaluation**
   - The extracted features are fed into a classifier to distinguish between normal and diseased kidneys.
   - Performance metrics include **Accuracy, Sensitivity, Specificity, and Area Under the Curve (AUC)**.
   - Evaluation results:
     - **VGG16:** Training Accuracy: 97%, Validation Accuracy: 99%
     - **ResNet50:** Training Accuracy: 100%, Validation Accuracy: 100%

## Installation and Requirements
To run this project, install the necessary dependencies:
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn opencv-python
```

## Running the Project
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd kidney-disease-detection
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open the provided `.ipynb` file and execute the cells step by step.

## Model Performance
- Confusion matrices and classification reports are used to evaluate model effectiveness.
- ResNet50 outperformed VGG16 in this task with higher accuracy and lower misclassification rates.
- The project demonstrates the potential of deep learning in medical imaging diagnostics.

## Future Improvements
- Enhancing the dataset with more labeled images for better generalization.
- Implementing additional pre-trained models like EfficientNet and DenseNet.
- Improving misclassification of tumor cases by refining model architectures.

## License
This project is open-source and available for educational and research purposes.

---
**Note:** Ensure compliance with medical data privacy and ethical considerations when using this model in real-world applications.

