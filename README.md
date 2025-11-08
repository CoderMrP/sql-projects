🧠 Breast Cancer Detection using Machine Learning
This project applies a Random Forest Classifier to detect whether a breast tumor is malignant (cancerous) or benign (non-cancerous) using diagnostic data from cell nuclei.
Built and tested in Google Colab using Python, Scikit-Learn, and Matplotlib.


📂 Dataset
Source: Breast Cancer Wisconsin (Diagnostic) Dataset
Shape: 569 rows × 30 features
Each row: One patient’s tumor sample
Each column: A numerical measurement (radius, texture, smoothness, concavity, etc.)
Target column:
0 = Malignant
1 = Benign


⚙️ Workflow
Data Loading & Exploration
Checked shape, null values, and data types
Data Cleaning & Preprocessing
Dropped unnecessary columns
Separated features (X) and target (y)
Split into training (80%) and test (20%) sets
Scaled features using StandardScaler
Model Training
Algorithm: RandomForestClassifier(random_state=42)
Trained on X_train, y_train
Model Evaluation
Accuracy on unseen data: 95.6%
Metrics: accuracy score, classification report, confusion matrix
Visualization
Test accuracy bar chart (Matplotlib)
Feature-importance ranking (top 10 features)


📊 Results
Metric	Score
Training Accuracy	1.00
Test Accuracy	0.956
Key Features	mean radius, mean concavity, area mean


🔍 Feature Importance Insights
After training the Random Forest Classifier, I analyzed the feature importance to understand which tumor characteristics influenced the model’s predictions the most.
The results showed that tumor size and shape irregularity — such as mean radius, mean area, and concavity worst — played a crucial role in determining malignancy.
This analysis highlighted how Random Forest can not only deliver high accuracy (95.6%) but also provide meaningful interpretability, which is essential for real-world medical applications.


🧩 Key Learnings
Difference between training vs test performance
Impact of standardization and feature scaling
Why Random Forest is robust against overfitting
How to visualize model results using Matplotlib


🛠️ Tech Stack
Language: Python
Libraries: Pandas | NumPy | Scikit-Learn | Matplotlib
Environment: Google Colab
🧬 Future Work: Image-Based Cancer Detection
Plan to extend this project by integrating image processing and deep learning (CNNs) to classify cancerous cells directly from microscopic images.
This will involve using OpenCV, TensorFlow, and Keras, enabling end-to-end automation from image input to diagnosis.


💬 Author
Paras Saini
🎓 MSc Data Analytics | Berlin
