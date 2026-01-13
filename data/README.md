# Public Health Risk Prediction System

## 🚀 Overview
This project predicts **public health risk** using real-world health indicators.
It includes:
✔ Machine Learning Model  
✔ Flask REST API  
✔ Streamlit UI  
✔ Data pipeline & feature engineering  

## 🧠 ML Pipeline
1. Raw CSV → cleaned
2. Feature extraction (15 inputs)
3. Risk label creation
4. Random Forest model trained
5. Model exported as `best_model.pkl`

## 🛠 Tech Used
- Python 3.11+
- Pandas / NumPy
- Scikit-learn
- Flask
- Streamlit

## 🏗 Project Structure
.
├── data/ (ignored)
├── model/ (contains training script)
│ ├── train.py
├── app/
│ ├── app.py (API)
│ ├── ui.py (Web UI)
├── requirements.txt
└── README.md

## ▶ How to Run

### 1️⃣ Train the model
python model/train.py
### 2️⃣ Start API
python app/app.py

nginx
Copy code
API runs at:  
http://127.0.0.1:5000/predict

### 3️⃣ Start UI
streamlit run app/ui.py

markdown
Copy code

UI runs at:  
http://localhost:8501/

## 📅 Next Goals
- Add feedback logging
- User authentication
- Deploy API + UI in cloud
- Convert to microservices
- Dashboard + history tracking

## 🙌 Author
Bhavina Parmar  
Public Health Risk ML – Startup Vision Project