# ❤️ Heart Disease Prediction MLOps Project

## 📌 Project Overview

This project is an end-to-end Machine Learning Operations (MLOps) pipeline for predicting the presence of heart disease using machine learning. The project demonstrates the complete workflow from data versioning to model deployment using industry-standard MLOps tools.

---

## 🎯 Objectives

- Predict whether a patient has heart disease.
- Track datasets using DVC.
- Train and compare multiple machine learning models.
- Track experiments using MLflow.
- Register the best-performing model.
- Deploy the model using FastAPI.
- Containerize the application using Docker.
- Automate the workflow using GitHub Actions.

---

## 🛠️ Technologies Used

- Python 3.9
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- MLflow
- DVC
- Docker
- Git & GitHub
- GitHub Actions

---

## 📂 Project Structure

```
HeartDisease_MLOps/
│
├── data/
│   └── heart.csv
│
├── models/
│   └── model.pkl
│
├── src/
│   ├── train.py
│   └── app.py
│
├── mlruns/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

- **Dataset:** Heart Disease Dataset
- **Records:** 1025
- **Features:** 13
- **Target:** Heart Disease (0 = No Disease, 1 = Disease)

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

The best-performing model is automatically saved as:

```
models/model.pkl
```

---

## 📈 Experiment Tracking

MLflow is used to:

- Log model parameters
- Track model accuracy
- Compare experiments
- Register the best model

Registered Model:

```
HeartDiseaseModel
```

---

## 🚀 FastAPI Deployment

Run the application:

```bash
uvicorn src.app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

Swagger UI allows testing the prediction API.

---

## 📦 Docker

Build Docker Image

```bash
docker build -t heart-disease-mlops .
```

Run Container

```bash
docker run -p 8000:8000 heart-disease-mlops
```

Access API

```
http://localhost:8000/docs
```

---

## 🔄 GitHub Actions

GitHub Actions automatically:

- Installs dependencies
- Builds the project
- Trains the model

Workflow file:

```
.github/workflows/ci.yml
```

---

## 📌 Installation

Clone the repository

```bash
git clone https://github.com/rheajenifer/Heart-Disease-MLOps-Project.git
```

Go to project directory

```bash
cd Heart-Disease-MLOps-Project
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run training

```bash
python src/train.py
```

Run API

```bash
uvicorn src.app:app --reload
```

---

## 📷 Screenshots
<img width="1917" height="817" alt="Screenshot 2026-08-07 235947" src="https://github.com/user-attachments/assets/b983ea27-c622-4729-947b-b4f9aa55b42d" />
<img width="951" height="781" alt="Screenshot 2026-08-08 020307" src="https://github.com/user-attachments/assets/b87c5155-c2a5-4898-8d73-2b510b76bed8" />
<img width="1807" height="976" alt="Screenshot 2026-08-08 000933" src="https://github.com/user-attachments/assets/6345c6bc-afe6-4ad3-923b-44d7ded34ddb" />
<img width="1890" height="913" alt="Screenshot 2026-08-08 000511" src="https://github.com/user-attachments/assets/b70278f2-41f8-45f4-97e1-15b5a00b2d92" />


---

## 👩‍💻 Author

**Rhea Jenifer P**

Third Year Computer Engineering Student

SSN College of Engineering

GitHub:
https://github.com/rheajenifer

---

## 📜 License

This project is created for educational and learning purposes.

---

## ⭐ Features

- End-to-End MLOps Pipeline
- Data Versioning using DVC
- Experiment Tracking using MLflow
- Model Registry
- REST API using FastAPI
- Docker Containerization
- GitHub CI/CD
- Machine Learning Model Comparison
