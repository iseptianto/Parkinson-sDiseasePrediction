# Parkinson's Disease Prediction using Machine Learning

[](https://opensource.org/licenses/MIT)
[](https://www.python.org/downloads/)
[](https://scikit-learn.org/)

This project is a machine learning implementation to detect Parkinson's disease based on the analysis of voice (vocal) features. A classification model is built to distinguish between healthy individuals and those with Parkinson's.

-----

### 📝 Description

Parkinson's disease is a neurodegenerative disorder that affects the nervous system and the parts of the body coordinated by the nerves. Early detection is crucial for management and treatment. This project utilizes voice measurement data, which is a non-invasive indicator, to train a machine learning model. The goal is to create a diagnostic aid that is fast, efficient, and accurate.

### 🎯 Problem Background

Conventional diagnosis of Parkinson's disease requires extensive and sometimes costly neurological examinations. The use of machine learning on voice data offers a more accessible and objective alternative. An accurate model can assist medical professionals in the initial screening process, enabling faster intervention and improving patients' quality of life.

### ✨ Key Features

  - **Exploratory Data Analysis (EDA)**: Visualization and statistical analysis to understand the distribution and correlation of data features.
  - **Data Preprocessing**: Application of feature scaling techniques (StandardScaler) to normalize the data range.
  - **Classification Model**: Utilizes a powerful model such as **XGBoost** or **Support Vector Machine (SVM)** for high performance.
  - **Model Evaluation**: Analyzes model performance with classification metrics like Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

### 📊 Dataset

The model is trained using the "Parkinson's Disease Dataset" from the UCI Machine Learning Repository, which is also available on Kaggle. This dataset consists of various biomedical voice measurements from 31 people, 23 of whom have Parkinson's disease.

  - **Source**: [UCI Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
  - **Number of Features**: 22 vocal features (e.g., MDVP:Fo(Hz), Jitter(%), Shimmer, etc.).
  - **Target Variable**: `status` - (1 for Parkinson's, 0 for healthy).

### 🛠️ Tech Stack

  - **Language**: Python 3.8+
  - **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost

### 🚀 Installation and Usage

1.  **Clone this repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate | macOS/Linux: source venv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebook:**

    ```bash
    jupyter notebook notebooks/parkinson_prediction_analysis.ipynb
    ```

### 📁 Project Structure

```
.
├── data/
│   └── parkinsons.csv
├── notebooks/
│   └── parkinson_prediction_analysis.ipynb
├── models/
│   └── xgboost_model.pkl
├── requirements.txt
└── README.md
```

### 🤖 Model and Evaluation

The **XGBoost Classifier** model was chosen for its superior performance on tabular datasets and its ability to handle class imbalance. All numerical features were scaled using `StandardScaler` to ensure optimal model performance.

The primary evaluation metrics used are:

  - **Accuracy**: The percentage of correctly classified predictions.
  - **Precision**: The model's ability to not mislabel a healthy subject as having Parkinson's.
  - **Recall (Sensitivity)**: The model's ability to find all subjects who actually have Parkinson's.
  - **F1-Score**: The harmonic mean of Precision and Recall.

The model achieves high accuracy and F1-Scores (e.g., \>90%), demonstrating its effectiveness in detecting Parkinson's disease from voice features.

### 🤝 Contributing

Contributions are welcome\! Please **Fork** this repository, create a new **Branch** for your feature, **Commit** your changes, and open a **Pull Request**.

### 📄 License

This project is licensed under the **MIT License**.
