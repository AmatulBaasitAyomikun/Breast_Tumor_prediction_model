# Breast Cancer Classification — Malignant vs. Benign Tumor Prediction

A machine learning classification project that predicts whether a breast tumor is **malignant** or **benign** based on measurements extracted from digitized images of fine needle aspirate (FNA) biopsies. The trained model is wrapped in an interactive Streamlit web app for real-time diagnosis support.

## Problem Statement

Breast cancer is one of the most common cancers affecting people worldwide, and early, accurate diagnosis is critical to effective treatment and improved survival rates. Traditional diagnosis relies on manual interpretation of biopsy characteristics, which can be time-consuming and subject to variability between practitioners.

This project explores whether a machine learning model can reliably classify a tumor as malignant or benign using quantitative cell nucleus features — offering a fast, consistent, data-driven second opinion that could support clinical decision-making.

## Solution

A **Logistic Regression** classifier was trained on the Breast Cancer Wisconsin (Diagnostic) dataset to distinguish malignant from benign tumors using 30 numeric features describing the size, shape, and texture of cell nuclei present in the biopsy image.

The end-to-end pipeline covers:

1. **Data Collection** — loaded the Breast Cancer Wisconsin dataset (via `sklearn.datasets` / provided as `Breast_Dataset.csv`).
2. **Data Exploration & Preprocessing** — checked dataset shape, structure, and null values; confirmed the target class balance (benign vs. malignant).
3. **Feature/Target Split** — separated the 30 predictive features from the diagnosis label.
4. **Train/Test Split** — stratified 80/20 split to preserve class balance across sets.
5. **Model Training** — trained a Logistic Regression model on the training set.
6. **Evaluation** — measured accuracy on both training and test data.
7. **Prediction System** — built a reusable function to classify new, unseen tumor measurements.
8. **Deployment** — serialized the model with `pickle` and deployed it behind a Streamlit UI for interactive predictions.

## Dataset

The **Breast Cancer Wisconsin (Diagnostic) dataset** contains 569 samples, each with 30 real-valued features computed from a digitized image of a breast mass, plus a diagnosis label.

- **Samples:** 569
- **Features:** 30 (10 base measurements × 3 statistics each: mean, standard error, and "worst"/largest value)
- **Target:** `diagnosis` — Malignant (M) or Benign (B)

**Base measurements per cell nucleus:**

| Feature | Description |
|---|---|
| radius | Mean distance from center to points on the perimeter |
| texture | Standard deviation of gray-scale values |
| perimeter | Nucleus perimeter |
| area | Nucleus area |
| smoothness | Local variation in radius lengths |
| compactness | perimeter² / area − 1.0 |
| concavity | Severity of concave portions of the contour |
| concave points | Number of concave portions of the contour |
| symmetry | Symmetry of the nucleus |
| fractal dimension | "Coastline approximation" − 1 |

Each of these is captured as a `_mean`, `_se` (standard error), and `_worst` value, producing the 30 total features.

## Model Performance

| Metric | Score |
|---|---|
| Training Accuracy | 94.7% |
| Testing Accuracy | 95.6% |

The model generalizes well, with test accuracy slightly exceeding training accuracy — indicating no overfitting on this dataset.

## Tech Stack

- **Language:** Python
- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** scikit-learn (Logistic Regression)
- **Deployment:** Streamlit, Pickle

## Repository Structure

```
├── Breast_Cancer_Classification_Project.ipynb   # EDA, preprocessing, training & evaluation
├── Breast_Dataset.csv                            # Breast Cancer Wisconsin (Diagnostic) dataset
├── Breast_cancer_prediction_web_app.py           # Streamlit app for live predictions
├── trained_model.sav                             # Serialized Logistic Regression model
└── README.md
```

## Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. Install dependencies:
   ```bash
   pip install streamlit numpy pandas scikit-learn
   ```

3. Update the model path in `Breast_cancer_prediction_web_app.py` — it currently points to a local, machine-specific path (`C:/Users/DELL PC/...`). Change it to a relative path, e.g.:
   ```python
   loaded_model = pickle.load(open('trained_model.sav', 'rb'))
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run Breast_cancer_prediction_web_app.py
   ```

5. Enter the 30 tumor measurement values in the form and click **Confirm Breast Tumor Type** to get a prediction.

## Example Prediction

Given a sample set of 30 tumor measurements, the model correctly classifies the tumor — e.g. an input resembling a small, regularly-shaped nucleus profile is classified as **Benign**, while irregular, larger measurements are classified as **Malignant**.

## Known Issues / Next Steps

- **Web app bug:** the prediction function is currently called with 30 separate positional arguments instead of a single list/array, which will raise a `TypeError` at runtime. This should be fixed to pass all inputs as one list, e.g. `breast_tumor_prediction([radius_mean, texture_mean, ...])`.
- **Hardcoded local path:** the model is loaded from an absolute Windows path — this needs to be relative for portability and deployment.
- Add input validation and sensible default/example values so users aren't submitting all-zero inputs by default.
- Explore additional models (Random Forest, SVM, XGBoost) and compare against the Logistic Regression baseline.
- Add a confusion matrix, precision/recall, and ROC-AUC to the evaluation for a fuller picture beyond accuracy — especially important in a medical context where false negatives carry high cost.
- Deploy the app publicly (e.g. Streamlit Community Cloud) and link it here.

## Disclaimer

This project is for educational and portfolio purposes only. It is **not** a certified diagnostic tool and should not be used for actual clinical decision-making.

## About This Project

This project was built to demonstrate an end-to-end machine learning workflow — from raw biomedical data to a deployed, interactive prediction tool — applied to a real, high-stakes healthcare problem. It reflects a broader interest in using data science to support better healthcare outcomes, particularly in resource-constrained or high-need contexts.
