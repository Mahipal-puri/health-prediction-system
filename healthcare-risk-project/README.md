# Smart Healthcare Risk Prediction System

A complete Data Science and Neural Network project that predicts whether a patient is **High Risk** or **Low Risk** using health parameters from the Medical Cost Personal Dataset.

## Project Overview

This project combines:
- **Data Science analysis** — cleaning, statistics, ANOVA, correlation, visualization
- **Neural Network model** — MLP Classifier for risk classification
- **Linear Regression** — for medical cost prediction
- **Interactive Streamlit frontend** — real-time patient risk prediction

## Dataset used

**Medical Cost Personal Dataset** from Kaggle containing 1,338 records with features:
- `age` — Patient age
- `sex` — Male / Female
- `bmi` — Body Mass Index
- `children` — Number of dependents
- `smoker` — Smoking status
- `region` — US geographic region
- `charges` — Medical insurance cost

**Risk Classification Rule:** charges > $15,000 → High Risk, otherwise Low Risk

## Project Structure

```
healthcare-risk-project/
├── dataset/
│   └── insurance.csv          # Raw dataset
├── notebooks/
│   └── data_analysis.ipynb    # Complete analysis notebook
├── model/
│   ├── train_model.py         # Model training script
│   ├── mlp_model.pkl          # Trained MLP classifier (after training)
│   ├── lr_model.pkl           # Linear regression model (after training)
│   ├── scaler.pkl             # Feature scaler (after training)
│   └── feature_cols.pkl       # Feature column names (after training)
├── frontend/
│   └── app.py                 # Streamlit web application
├── requirements.txt
└── README.md
```

## Setup & Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the models
python model/train_model.py

# 3. Launch the Streamlit app
streamlit run frontend/app.py
```

## Syllabus Coverage

### Data Science Lab
| Topic | Implementation |
|-------|---------------|
| Reading files with open() | CSV file reading |
| Pandas data loading | pd.read_csv() |
| Dataset understanding | info(), describe() |
| Missing value handling | fillna() |
| Data formatting | Label encoding |
| Normalization | MinMaxScaler, StandardScaler |
| Binning | BMI categories, age groups |
| Indicator variables | One-Hot Encoding (region) |
| Descriptive statistics | describe(), skew(), kurtosis() |
| ANOVA | f_oneway() tests |
| Correlation | corr(), heatmap |
| Linear regression | Charges prediction |
| Visualization | Histogram, scatter, pie, bar, box plots |

### Neural Network Lab
| Topic | Implementation |
|-------|---------------|
| Perceptron | Basic NN classifier concept |
| Multi-layer feed-forward network | MLPClassifier (8-4 architecture) |
| Line separation | Decision boundary visualization |
| 2D cluster classification | Scatter plot with risk classes |
| Classification | Patient risk classification |

## Technologies

- **Python** — Programming language
- **Pandas** — Data manipulation
- **NumPy** — Numerical computing
- **Matplotlib & Seaborn** — Visualization
- **Scikit-learn** — Machine learning & neural networks
- **SciPy** — Statistical tests
- **Streamlit** — Interactive web frontend
- **Folium** — Geospatial visualization
