# HR Employee Attrition Predictor

An end-to-end machine learning system that predicts employee attrition risk with 74.8% accuracy and provides actionable HR recommendations.

**Live Demo:** [Click Here to Try the App](https://hr-employee-attrition-predictor.streamlit.app/)

**Google Colab Notebook:** [Click Here to View the Code](https://colab.research.google.com/drive/1Fuai9H5xwgR6oZk6ylKCCDDA_MCN28AV?usp=sharing)

---

## 📌 What is This Project?

Every company loses employees, but losing the wrong employees at the wrong time costs heavily in hiring, training, and lost productivity. This project solves a critical HR problem: **predicting which employees are likely to leave** before they actually quit.

For HR teams, this means:
- Proactive retention strategies
- Reduced hiring and training costs
- Data-driven employee engagement
- Targeted interventions for high-risk employees

This is a complete data science pipeline from raw HR data to an interactive web application.

---

## 🛠️ Technology Stack

| Category | Tools & Libraries |
|----------|-------------------|
| **Programming Language** | Python 3.x |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting) |
| **Model Serialization** | Joblib |
| **Web Framework** | Streamlit |
| **Deployment** | Streamlit Cloud |
| **Version Control** | GitHub |

---

## 💰 Currency Support

This project displays predictions and salary information in **both** currencies:

| Currency | Symbol | Conversion |
|----------|--------|------------|
| US Dollar | $ | Original dataset currency |
| Indian Rupee | ₹ | For local context ($1 ≈ ₹83) |

**Example:** Monthly income of $5,000 is approximately ₹4,15,000.

---

## 🚀 How It Works

### 1. Data Loading & Exploration
- Loaded 1,470 employee records with 35 features
- No missing values found
- Target column: Attrition (Yes/No)
- Attrition rate: 16.12% (imbalanced dataset)

### 2. Data Cleaning & Preprocessing
- Dropped non-predictive columns: `EmployeeNumber`, `Over18`, `StandardHours`, `EmployeeCount`
- Converted target from Yes/No to 1/0
- One-hot encoding for categorical variables
- Standard scaling for numeric features
- Final dataset: 1,470 rows, 44 features

### 3. Exploratory Data Analysis
- Sales department has highest attrition at 20.6%
- Sales Representative role has highest attrition at 39.8%
- Employees who left earn $2,045 less on average
- Employees with WorkLifeBalance rating 1 have 31.3% attrition rate
- Attrition peaks at certain tenure milestones

### 4. Model Building
Trained and compared three models with `class_weight='balanced'`:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 74.8% | 34.1% | 61.7% | 43.9% | 79.9% |
| Random Forest | 83.3% | 37.5% | 6.4% | 10.9% | 75.2% |
| Gradient Boosting | 85.0% | 58.8% | 21.3% | 31.3% | 79.4% |

**Best Model:** Logistic Regression (based on F1-Score and Recall)

### 5. Feature Importance (Top 5)

| Feature | Direction |
|---------|-----------|
| JobRole_Laboratory Technician | Increases attrition |
| OverTime_Yes | Increases attrition |
| BusinessTravel_Travel Frequently | Increases attrition |
| JobSatisfaction | Decreases attrition |
| WorkLifeBalance | Decreases attrition |

### 6. Deployment
The trained model is saved as `attrition_model.joblib` and served through a Streamlit web app with an interactive HR simulator.

---

## 🧠 Challenges Faced & Solutions

| Challenge | Solution |
|-----------|----------|
| **Class Imbalance** | Used `class_weight='balanced'` parameter in models |
| **Feature Mismatch in App** | Aligned all 44 features in the prediction pipeline |
| **Encoding Categorical Variables** | Used `pd.get_dummies()` with `drop_first=True` |
| **Model Serialization** | Used `joblib` for consistent cross-version compatibility |
| **Cloud Deployment** | Added `requirements.txt` to ensure correct library versions |
| **Entering Multiple Employee Data** | Built an interactive form with sliders and dropdowns for easy data entry |

### Code Breaking in `app.py`
**Problem:** The model expected 44 features, but the app was sending incomplete data.
**Fix:** Created a complete feature vector with all 44 features, using default values for non-critical features.

---

## 💼 Business Value & Scalability

### Real-World Applications:
1. **HR Teams:** Identify high-risk employees before they leave
2. **Managers:** Proactively engage with at-risk team members
3. **Recruitment:** Reduce hiring costs by retaining talent
4. **Leadership:** Data-driven workforce planning

### Scalability:
- **More Data:** Can incorporate additional employee records
- **More Features:** Can add performance reviews, engagement survey data
- **API Integration:** Can be deployed as a REST API for HR systems
- **Real-Time Monitoring:** Can integrate with HRIS for continuous prediction

---

## 🎯 Unique Selling Points (USPs)

| USP | Why It Matters |
|-----|----------------|
| **74.8% Accuracy** | Reliable predictions for HR decision-making |
| **Interactive Simulator** | HR teams can test scenarios without coding |
| **Actionable Recommendations** | Specific steps HR can take immediately |
| **Dual Currency** | USD and INR for global and local context |
| **Explainable AI** | Feature importance reveals what drives attrition |
| **End-to-End Pipeline** | From data to deployed app in one workflow |

---

## ⚠️ Limitations & Future Improvements

| Limitation | How to Overcome |
|------------|-----------------|
| **Limited Features** | Add more predictors: engagement scores, peer reviews, promotion history |
| **Single Company Data** | Train on data from multiple companies |
| **Static Model** | Implement auto-retraining with new data |
| **No Time Dimension** | Add time-series analysis to track changes over time |
| **Binary Classification** | Could predict severity or time-to-leave instead of just Yes/No |

### Room for Improvement:
1. **Time-Series Analysis** - Track employee sentiment over time
2. **Customizable Thresholds** - Let HR adjust risk sensitivity
3. **Email Alerts** - Notify HR when high-risk employees are identified
4. **Dashboard Analytics** - Show trends and patterns over time
5. **A/B Testing** - Test intervention effectiveness


## 📂 Project Structure
