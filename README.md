
# 📊 Customer Retention Prediction - Telecom Industry

A complete end-to-end machine learning project to analyze and predict customer retention in the telecom sector. This solution leverages **exploratory data analysis**, **feature engineering**, **gradient boosting models**, and a fully interactive **Streamlit dashboard** to deliver real-time retention risk predictions.

---

## 🚀 Project Overview

Customer retention refers to when a customer discontinues their service. Identifying the risk of retention helps telecom companies take proactive steps to improve retention. In this project:

- We analyzed real-world customer data to uncover patterns and key indicators of retention.
- Trained a **Gradient Boosting Classifier** for high accuracy predictions.
- Built a modern **Streamlit web app** for instant retention prediction based on customer attributes.

---

## 🎯 Key Features

- **Data Exploration & Visualization**  
  Understand customer behavior and retention trends using rich visual analysis.

- **ML Pipeline with Gradient Boosting**  
  Final model trained with robust tuning for optimal performance.

- **Streamlit Dashboard**  
  User-friendly web app interface to input customer details and receive churn predictions.

- **Probability-Based Insights**  
  Shows confidence of the prediction with interactive bar charts powered by Plotly.

---

## 🛠️ Tech Stack

- **Python**, **Pandas**, **NumPy**, **Scikit-learn**
- **Gradient Boosting Classifier** (via Scikit-learn)
- **Streamlit** for frontend UI
- **Plotly** for dynamic visualizations
- **Joblib** for model serialization

---

## 🔍 How to Run the App

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Enter customer details in the UI and receive predictions with probability.

---

## 📈 Model Inputs

- Demographic: Gender, Senior Citizen, Partner, Dependents  
- Services: Internet, Phone, Streaming, Support  
- Financial: Monthly Charges, Total Charges, Tenure  
- Contract Info: Contract Type, Payment Method, Paperless Billing

---

## 🧠 Insights

- Customers with **month-to-month contracts**, **fiber optic service**, and **high monthly charges** showed higher retention rates.
- Our model achieves strong predictive performance with explainable features.

---

## 🏁 Future Enhancements

- Integrate SHAP for explainable AI insights  
- Add database support for real-time predictions  
- Deploy on cloud (AWS/GCP/Heroku)

---

## 💡 Developed With

**Streamlit**, **Scikit-learn**, and a passion for data-driven decision making 🚀
