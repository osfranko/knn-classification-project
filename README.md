# K-Nearest Neighbors Breast Cancer Diagnosis

**Pattern Recognition Mini Project**

---

## Project Overview

This project implements a K-Nearest Neighbors (KNN) classifier to accurately diagnose breast tumors as malignant or benign. KNN is a simple yet powerful instance-based learning algorithm that makes predictions based on the majority class of the k closest training examples in the feature space. 

The project follows a complete machine learning workflow — from data exploration to model evaluation and real-world prediction — while emphasizing the critical role of feature scaling and hyperparameter tuning in distance-based algorithms.

---

## The Dataset

The Breast Cancer Wisconsin (Diagnostic) dataset contains 569 instances with 30 real-valued numerical features. These features are computed from digitized images of fine needle aspirates (FNA) of breast masses and describe characteristics of cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension (with mean, standard error, and worst values).

The target variable is binary:
- 0 → Malignant tumor
- 1 → Benign tumor

The dataset is moderately imbalanced with approximately 37.3% malignant and 62.7% benign cases. No missing values were present.

---

## Exploratory Data Analysis Observations

During exploration, we observed that the features had significantly different scales and ranges. For example, `mean area` ranged from 143 to 2501, while `mean smoothness` ranged between 0.05 and 0.16. Histograms showed that most features were right-skewed, which is typical in medical measurement data.

Boxplots further confirmed the wide variation in feature magnitudes. This analysis clearly indicated that feature scaling would be essential for KNN, as unscaled data would allow large-range features to dominate distance calculations.

---

## Methodology & What Was Done

We split the data into 80% training and 20% testing sets using stratification to maintain class distribution. Features were then normalized using `StandardScaler`, which transformed all features to have zero mean and unit variance. This step ensured equal contribution from all features in distance computations.

A baseline KNN model was trained with k=5. We then performed hyperparameter tuning by testing k values (1, 3, 5, 7, 9, 11) using 5-fold cross-validation on the training set. The value **k=3** produced the highest cross-validation accuracy of 96.92%.

The final model was trained with k=3 and evaluated on the unseen test set. We also tested the model on new synthetic patient samples to demonstrate practical usability.

---

## Key Results & Model Evaluation

The final KNN model (k=3) achieved an outstanding **test accuracy of 98.25%**.

**Detailed Classification Report:**
- **Malignant (0)**: Precision = 1.00, Recall = 0.95, F1-Score = 0.98
- **Benign (1)**: Precision = 0.97, Recall = 1.00, F1-Score = 0.99

The confusion matrix showed only **2 False Negatives** and **0 False Positives** out of 114 test samples. This is a strong result for medical diagnostics, where correctly identifying malignant cases is critical.

When tested on new sample data points, the model made confident and correct predictions (100% confidence on both test cases).

---

## Key Learnings

This project reinforced several important concepts in machine learning:
- Feature scaling is crucial for KNN. Without it, features with larger numerical ranges would disproportionately influence distance calculations, leading to poor performance.
- Cross-validation provides a more reliable way to select hyperparameters than using a single train-test split.
- In medical applications, evaluating models with multiple metrics (especially Precision, Recall, and Confusion Matrix) is more insightful than relying solely on accuracy.
- Even a simple algorithm like KNN can deliver excellent results when the data is properly preprocessed and the hyperparameter (k) is well-tuned.

---

## Technologies Used

- Python
- scikit-learn (for KNN, scaling, splitting, and evaluation)
- pandas & numpy (data manipulation and analysis)
- matplotlib & seaborn (data visualization)

---

## Project Structure

- `breast_cancer_diagnosis.ipynb` – Complete Jupyter Notebook with all steps and visualizations
- `README.md` – Project documentation
- `requirements.txt` – Required Python dependencies

---

## Setup Instructions

To run this project on your local machine:

```bash
git clone https://github.com/osfranko/knn-classification-project.git
cd knn-classification-project
pip install -r requirements.txt
jupyter notebook breast_cancer_diagnosis.ipynb
Author: osfranko
Project Date: June 2026
