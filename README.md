# Uncertainty-Aware Ensemble Learning for Loan Default Prediction

An end-to-end **Artificial Intelligence and Machine Learning** project that predicts loan default probability and estimates predictive uncertainty using ensemble learning techniques.

The system combines multiple machine learning algorithms, Soft Voting Ensembles, Bootstrap Resampling, Inter-Model Disagreement, and Predictive Shannon Entropy to provide more informative loan risk predictions.

> **Educational Prototype Disclaimer:** This is a B.Tech engineering educational prototype developed using a synthetic demonstration dataset. It is not a validated real-world lending or underwriting decision system.

---

## 1. Project Overview

Traditional loan prediction systems generally provide a simple result, such as Default or No Default.

This project goes one step further by estimating how reliable a prediction is.

It provides:
- Loan default probability.
- Prediction confidence.
- Uncertainty score.
- Bootstrap confidence interval.
- Explanations for uncertain predictions.

The objective is to support better understanding of financial risk through Artificial Intelligence and Machine Learning.

## 2. Key Features

### Machine Learning Algorithms

The project implements six machine learning approaches:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost
6. Soft Voting Ensemble

### Uncertainty Estimation

The system uses multiple techniques:

- **Bootstrap Resampling:** Evaluates prediction variation using repeated samples.
- **Inter-Model Disagreement:** Measures differences between machine learning model predictions.
- **Predictive Shannon Entropy:** Measures uncertainty in predicted probabilities.
- **Confidence Interval:** Provides a range of possible prediction values.

### Data Preprocessing

- Missing value handling using median and mode imputation.
- Outlier handling using IQR-based winsorization.
- Debt-to-Income (DTI) ratio calculation.
- Loan-to-Income (LTI) ratio calculation.
- One-hot encoding.
- Standard scaling.

### Additional Features

- Loan default risk prediction.
- Prediction confidence and uncertainty scores.
- SQLite database storage.
- Prediction history.
- HTML prediction reports.
- CSV report export.
- Model evaluation and visualization.

## 3. Technologies Used

| Category | Technology |
|---|---|
| Programming Language | Python |
| Web Framework | Flask |
| Machine Learning | Scikit-learn |
| ML Algorithm | XGBoost |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |

## 4. Project Architecture

The project follows this workflow:

1. Applicant enters loan information.
2. Flask receives the applicant details.
3. Data preprocessing prepares the input.
4. Machine learning models generate predictions.
5. Soft Voting Ensemble combines model probabilities.
6. Uncertainty estimation evaluates prediction reliability.
7. Final results are displayed on the web application.
8. Applicant details and prediction results can be stored in SQLite.

## 5. Project Structure

```text
Loan-Default-Prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── generate_dataset.py
│   └── loan_data.csv
│
├── models/
│   └── Saved ML models
│
├── evaluation/
│   └── Evaluation results and graphs
│
├── database/
│   ├── schema.sql
│   └── db.py
│
├── templates/
│   └── HTML templates
│
├── static/
│   ├── css/
│   └── js/
│
├── docs/
└── tests/
```

*Note: This is the intended project structure. Keep only folders and files that actually exist in your repository.*

## 6. Installation and Execution

### Prerequisites

- Python 3.10 or later.
- pip package manager.
- Git (optional).

### Step 1: Clone the Repository

```bash
git clone https://github.com/Thaheer-prog/Uncertainty-aware-loan-default-prediction.git
```

### Step 2: Open the Project Folder

```bash
cd Uncertainty-aware-loan-default-prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

For Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Train the Machine Learning Models

```bash
python train_model.py
```

### Step 7: Run the Application

```bash
python app.py
```

Open the following address in your browser:

http://127.0.0.1:5000

## 7. Model Evaluation

The models are evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Reported Benchmark Results

Evaluation using an 80/20 stratified train-test split (241 test records):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 78.84% | 66.67% | 45.45% | 54.05% | 77.35% |
| Decision Tree | 80.50% | 67.92% | 54.55% | 60.50% | 79.19% |
| Random Forest | 80.50% | 73.17% | 45.45% | 56.07% | 80.71% |
| Gradient Boosting | 78.42% | 64.58% | 46.97% | 54.39% | 76.56% |
| XGBoost | 79.67% | 68.09% | 48.48% | 56.64% | 76.48% |
| Soft Voting Ensemble | 80.50% | 70.21% | 50.00% | 58.41% | 78.25% |

*These are the reported benchmark values for the demonstration dataset, not guaranteed results on other datasets.*

## 8. How Uncertainty Is Estimated

1. Each machine learning model generates its own default probability.
2. The Soft Voting Ensemble combines model probabilities.
3. Bootstrap resampling measures prediction variation.
4. Inter-model disagreement measures differences between model outputs.
5. Predictive entropy measures uncertainty in the probability distribution.
6. The system combines uncertainty information to identify predictions requiring additional attention.

A higher uncertainty score indicates that the prediction is less reliable.

## 9. Applications

- Banking and financial institutions.
- Credit risk assessment.
- Loan default analysis.
- Financial risk management.
- Educational machine learning research.

## 10. Future Enhancements

- Integration with real-world financial datasets.
- Improved model calibration.
- Advanced uncertainty estimation methods.
- Cloud deployment.
- Enhanced interactive dashboard.
- Additional machine learning algorithms.

## 11. Conclusion

This project demonstrates how Artificial Intelligence and Machine Learning can be used for loan default prediction.

By combining ensemble learning with uncertainty estimation, the system provides more information than a simple default or no-default prediction.

It demonstrates the importance of understanding both the prediction and the confidence associated with it.

