# Financial Market A/B Testing & Machine Learning

## Project Overview

This project implements an advanced analytical framework for evaluating financial market strategies using a combination of:

* Market segmentation (Clustering)
* Profitability prediction (Machine Learning)
* Bayesian A/B testing (Statistical inference)

The objective is to analyze and optimize trading strategies by comparing control and treatment groups in simulated financial market scenarios.

---

## Key Features

### 🔹 Market Segmentation

* Uses **K-Means clustering** to identify market regimes
* Groups data into segments based on behavior patterns

### 🔹 Profitability Prediction

* Logistic Regression model to predict trade success
* Handles class imbalance using `class_weight='balanced'`
* Evaluated using ROC AUC and classification report

### 🔹 Bayesian A/B Testing

* Implemented using **PyMC**
* Compares control vs treatment groups
* Estimates:

  * Difference in performance (delta)
  * Relative lift
* Uses MCMC sampling for probabilistic inference

### 🔹 Visualization

* Posterior distribution plots for decision-making
* Insights into statistical significance of strategies

---

## Dataset Description

The dataset simulates financial market conditions and includes:

| Feature       | Description                |
| ------------- | -------------------------- |
| price_change  | Price variation            |
| volume_change | Trading volume variation   |
| return_after  | Return after strategy      |
| is_profit     | Binary indicator of profit |
| group         | Control or Treatment       |
| ticker_num    | Asset identifier           |
| market_num    | Market segment             |

---

## Project Workflow

```text
Data Generation / Load
        ↓
Market Segmentation (KMeans)
        ↓
Profitability Prediction (Logistic Regression)
        ↓
Bayesian A/B Testing (PyMC)
        ↓
Visualization & Insights
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* PyMC
* Logging

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the script:

```bash
python mercadofinanciero_abtesting.py
```

---

## Output

* Market segments added to dataset
* Classification report (precision, recall, f1-score)
* ROC AUC score
* Bayesian posterior distributions
* Visual insights for decision-making

---

## Project Structure

text
project/
│
├── mercadofinanciero_abtesting.py
├── README.md
├── requirements.txt


---

## Use Cases

* Evaluate trading strategies
* Financial experimentation (A/B testing)
* Risk and performance analysis
* Behavioral finance research
* Algorithmic trading validation

---

## Future Improvements

* Add real financial datasets
* Deploy as API for real-time predictions
* Implement deep learning models
* Extend to multi-variant A/B testing
* Dashboard integration (Streamlit / Dash)

---

## Author
Flavia Hepp
Advanced Data Science project combining Machine Learning and Bayesian Statistics applied to financial markets.
