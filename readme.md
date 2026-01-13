📌 README.md — Customer Churn Prediction using ANN
# Customer Churn Prediction using Artificial Neural Network (ANN)

## 📖 Project Overview
Customer churn is one of the major challenges faced by banks and financial institutions.  
This project focuses on predicting whether a customer is **likely to leave the bank (churn)** or **stay loyal**, using an **Artificial Neural Network (ANN)** model trained on a churn dataset.

The project is divided into **three major parts**:
1. ANN Model Training  
2. Customer Churn Prediction  
3. Model Deployment using Streamlit  

This solution helps banks take **proactive decisions** to retain customers by identifying high-risk churn cases.

---

## 🧠 Problem Statement
To build a machine learning model that predicts customer churn based on demographic and financial details such as:
- Credit score
- Age
- Balance
- Tenure
- Number of products
- Active membership
- Estimated salary

---

## 🏗️ Project Structure


Customer-Churn-ANN/
│
├── data/
│ └── churn_dataset.csv
│
├── model/
│ └── ann_model.h5
│
├── notebooks/
│ └── ann_training.ipynb
│
├── app.py
├── requirements.txt
└── README.md


---

## ⚙️ Part 1: ANN Model Training
- Data preprocessing (encoding, scaling, feature selection)
- Splitting data into training and testing sets
- Building an ANN with:
  - Input layer
  - Hidden layers
  - Output layer (Binary Classification)
- Model evaluation using accuracy and loss
- Saving the trained model for deployment

---

## 🔮 Part 2: Customer Churn Prediction
- Takes customer input data
- Applies the same preprocessing used during training
- Predicts whether the customer will:
  - **Stay Loyal**
  - **Exit (Churn)**

---

## 🚀 Part 3: Model Deployment using Streamlit
- Interactive web interface for real-time prediction
- Users can input customer details using sliders and dropdowns
- Displays churn prediction instantly
- Makes the ML model accessible without technical knowledge

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Libraries & Frameworks:**
  - NumPy
  - Pandas
  - Scikit-learn
  - TensorFlow / Keras
- **Model Type:** Artificial Neural Network (ANN)
- **Web Framework:** Streamlit
- **IDE & Tools:** VS Code, Jupyter Notebook
- **Version Control:** Git & GitHub

---

## 📊 Dataset
- Publicly available **Bank Customer Churn Dataset**
- Contains customer demographic and financial information
- Binary target variable:
  - `1` → Customer Exited  
  - `0` → Customer Stayed

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/HemantKumar0505

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run Streamlit App
streamlit run app.py

📌 Future Enhancements

Hyperparameter tuning for better accuracy

Add model performance metrics in UI

Deploy the app on cloud platforms (Heroku / AWS / Streamlit Cloud)

Compare ANN with other ML models

👤 Author

Hemant Kumar
Btech ug | Python | SQL | AI & ML Enthusiast 

⭐ Acknowledgement

Thanks to open-source datasets and libraries that made this project possible.


