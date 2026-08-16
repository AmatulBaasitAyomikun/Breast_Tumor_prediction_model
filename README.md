# Breast Cancer Classification — Malignant vs. Benign Tumor Prediction

A machine learning classification project that predicts whether a breast tumor is **malignant** or **benign** based on measurements extracted from digitized images of fine needle aspirate (FNA) biopsies. The trained model is deployed behind an interactive Streamlit web app for real-time diagnosis support.

🎗️ **[Live Demo](https://breast-tumor-pred-app.streamlit.app/)**

## Problem Statement

Breast cancer is one of the most common cancers affecting people worldwide, and early, accurate diagnosis is critical to effective treatment and improved survival rates. Traditional diagnosis relies on manual interpretation of biopsy characteristics, which can be time-consuming and subject to variability between practitioners.

This project explores whether a machine learning model can reliably classify a tumor as malignant or benign using quantitative cell nucleus features — offering a fast, consistent, data-driven second opinion that could support clinical decision-making.

## Solution

A **Logistic Regression** classifier was trained on the Breast Cancer Wisconsin (Diagnostic) dataset to distinguish malignant from benign tumors using 30 numeric features describing the size, shape, and texture of cell nuclei present in the biopsy image. The model is served through a clean, sectioned Streamlit interface that groups inputs by feature type and returns a color-coded diagnosis.

The end-to-end pipeline covers:

1. **Data Collection** — loaded the Breast Cancer Wisconsin dataset (via `sklearn.datasets` / provided as `Breast_Dataset.csv`).
2. **Data Exploration & Preprocessing** — checked dataset shape, structure, and null values; confirmed the target class balance (benign vs. malignant).
3. **Feature/Target Split** — separated the 30 predictive features from the diagnosis label.
4. **Train/Test Split** — stratified 80/20 split to preserve class balance across sets.
5. **Model Training** — trained a Logistic Regression model on the training set.
6. **Evaluation** — measured accuracy on both training and test data.
7. **Prediction System** — built a reusable function that takes a single feature list and returns a diagnosis.
8. **Deployment** — serialized the model with `pickle` and deployed it behind a polished Streamlit UI, ready for cloud hosting.

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

## Web App Features

- 🎨 Clean, centered layout with a custom page title and icon
- 📊 Inputs organized into three logical sections — **Mean**, **Standard Error**, and **Worst** features — each laid out in columns for readability
- 🟢🔴 Color-coded, emoji-flagged diagnosis result for quick visual interpretation
- ⚡ Single-click prediction via a reusable, correctly-batched prediction function

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
├── breast_tumor_web_app.py                       # Streamlit app for live predictions
├── trained_model.sav                             # Serialized Logistic Regression model
├── requirements.txt                              # Python dependencies for deployment
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
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run breast_tumor_web_app.py
   ```

4. Open the local URL shown in your terminal, fill in the tumor measurements across the Mean, SE, and Worst sections, and click **Predict Tumor Type**.

## Deploying to Streamlit Community Cloud

This app is deployment-ready as-is:

1. Push this repository to GitHub, including `trained_model.sav` and `requirements.txt`.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repository and branch, and set the main file path to `breast_tumor_web_app.py`.
4. Click **Deploy**. Streamlit Cloud will install everything from `requirements.txt` automatically.

No code changes are required for deployment — the model is already loaded via a relative path (`trained_model.sav`), so it works the same locally and in the cloud.

## Example Prediction

Given a sample set of 30 tumor measurements, the model correctly classifies the tumor — e.g. an input resembling a small, regularly-shaped nucleus profile is classified as 🟢 **Benign**, while irregular, larger measurements are classified as 🔴 **Malignant**.

## Next Steps

- Explore additional models (Random Forest, SVM, XGBoost) and compare against the Logistic Regression baseline.
- Add a confusion matrix, precision/recall, and ROC-AUC to the evaluation for a fuller picture beyond accuracy — especially important in a medical context where false negatives carry high cost.
- Add input validation and example/preset values so first-time users aren't stuck entering 30 zeros.
- Add a short "how to read this result" note in the UI to make the tool more approachable for non-technical users.

## Disclaimer

This project is for educational and portfolio purposes only. It is **not** a certified diagnostic tool and should not be used for actual clinical decision-making.

## About This Project

This project was built to demonstrate an end-to-end machine learning workflow — from raw biomedical data to a deployed, interactive prediction tool — applied to a real, high-stakes healthcare problem. It reflects a broader interest in using data science to support better healthcare outcomes, particularly in resource-constrained or high-need contexts.
