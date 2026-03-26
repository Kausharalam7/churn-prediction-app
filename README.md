# 📊 Customer Churn Prediction System

## 🚀 Project Overview
This project predicts whether a customer will churn (leave) or stay using machine learning.

Customer churn prediction is important for businesses because retaining customers is more cost-effective than acquiring new ones.

---

## 🧠 Features
- Predict customer churn in real-time
- Handles class imbalance using SMOTE
- Uses pipeline to avoid data leakage
- Interactive web app using Streamlit

---

## ⚙️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit

---

## 📊 Model Details
- Algorithm: Random Forest Classifier
- Encoding: OneHotEncoding
- Evaluation: F1-score, ROC-AUC
- Pipeline used for preprocessing + modeling

---

## 🌍 Live Demo
👉 https://churn-prediction-app-baj4keuyr9ctmk5gsxetxd.streamlit.app/

---

## 📁 Project Structure
```
app.py        # Streamlit app
train.py      # Model training
model.pkl     # Trained model
churn.csv     # Dataset
requirements.txt

````

---

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
````

---

## 🎯 Future Improvements

* Add dashboard analytics
* Improve model accuracy
* Deploy using FastAPI + Docker

---

## 👨‍💻 Author

Kaushar Alam
