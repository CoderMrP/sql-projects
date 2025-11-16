🧠 Breast Cancer Detection using Machine Learning & Tableau Dashboard

This project builds a Random Forest Classifier to detect whether a breast tumor is Malignant (cancerous) or Benign (non-cancerous) using the Breast Cancer Wisconsin Diagnostic dataset.
The final predictions and evaluation metrics are visualized using an interactive Tableau Dashboard.


🔗 Live Dashboard
👉 View the Tableau Dashboard:
https://public.tableau.com/views/MLCanceerDetection/BreastCancerClassificationMLModelPerformanceDashboard?:language=en-GB&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link


📂 Dataset
Breast Cancer Wisconsin (Diagnostic)
569 rows × 30 features
Each row = one patient’s tumor sample
Each feature = cell nucleus measurements
Target values:
0 → Malignant
1 → Benign


⚙️ Project Workflow

1️⃣ Data Loading & Exploration
Checked dataset shape, structure, missing values
Viewed correlations & distributions

2️⃣ Data Cleaning & Preprocessing
Ignored unnecessary columns
Prepared features (X) and labels (y)
Train-test split (80% / 20%)
Standardized features using StandardScaler

3️⃣ Model Training
Algorithm: RandomForestClassifier(random_state=42)
Trained on scaled training data
Extracted feature importances

4️⃣ Model Evaluation
Evaluated using:
Accuracy
Precision
Recall
F1 Score
Confusion Matrix


📈 Model Performance
Metric	Score
Training Accuracy	100%
Test Accuracy	95.6%
Best CV Accuracy (GridSearchCV)	96.04%
Precision (Malignant)	95%
Recall (Malignant)	93%
Precision (Benign)	96%
Recall (Benign)	97%


🧠 Confusion Matrix (Test Results)
Predicted: Malignant	Predicted: Benign
Actual: Malignant	39	3
Actual: Benign	2	70
✔ Correctly identified 39/42 malignant tumors (93%)
✔ Correctly identified 70/72 benign tumors (97%)
✔ Balanced sensitivity & specificity → ideal for medical use cases


📊 Tableau Dashboard Overview
Visualized on Tableau Public with:
KPI Cards
Accuracy: 95.6%
Precision: 95%
Recall: 93%
F1 Score: 94%
Dashboard Visuals
Confusion Matrix
Prediction Distribution
Actual vs Predicted Comparison
Correct vs Incorrect Predictions


🔍 Feature Importance Insights
Random Forest identified key predictors:
mean radius
mean concavity
area mean
worst concavity
worst perimeter
These reflect tumor size, texture, and shape irregularity — medically meaningful indicators of malignancy.


🧩 Key Learnings
Difference between training vs test performance
Importance of feature scaling
Why Random Forest reduces overfitting
Creating professional dashboards in Tableau
Interpreting ML classification metrics


🧬 Future Work
1. Deep Learning for Image-Based Cancer Detection
Use CNNs (TensorFlow/Keras)
Detect cancer directly from microscopic images
Combine image data + numeric features
2. ROC & AUC Visualizations
Add advanced evaluation metrics to Tableau.
3. Streamlit Web App
Allow users to upload measurements and get predictions.


🛠 Tech Stack
Python (Pandas, NumPy, Scikit-Learn, Matplotlib)
Tableau Desktop + Tableau Public
Google Colab


👤 Author
Paras Saini
🎓 MSc Data Analytics, Berlin
🔗 LinkedIn: https://www.linkedin.com/in/paras-saini-0a64b7250
