# 🚢 Titanic Survival Prediction — End-to-End Machine Learning App

An end-to-end Machine Learning application built with **FastAPI** for the backend API, **Scikit-Learn** for model training & inference, and **HTML5/CSS3** with **Jinja2 Templates** for an interactive, user-friendly UI.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Getting Started](#-getting-started)
- [Model Training](#-model-training)
- [Running the Web Application](#-running-the-web-application)
- [API Endpoints](#-api-endpoints)
- [License](#-license)

---

## 📖 Overview
This application predicts whether a passenger would have survived the historic Titanic disaster based on features like passenger class, age, gender, fare, number of relatives aboard, and port of embarkation. 

The machine learning model is trained using a **Random Forest Classifier** on the classic Titanic dataset, achieving reliable accuracy, and served in real-time via FastAPI.

---

## ✨ Features
- **Interactive UI:** Clean and responsive user input form designed with HTML5 and CSS3.
- **FastAPI Integration:** Asynchronous, lightweight, and high-performance backend server.
- **Real-Time Predictions:** Calculates survival probability and displays confidence score instantly upon submission.
- **Modular Design:** Clear separation between data preprocessing/model training and API serving logic.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib
* **Backend Framework:** FastAPI, Uvicorn
* **Frontend UI:** HTML5, CSS3, Jinja2 Templating Engine

---

## 📁 Project Directory Structure

```text
titanic_project/
│
├── train.csv              # Training Dataset
├── test.csv               # Test Dataset
├── train_model.py         # Script to train and save the ML model
├── titanic_model.pkl      # Serialized trained Random Forest model
├── main.py                # FastAPI backend & routes handling
├── templates/
│   └── index.html         # Jinja2 HTML template for UI
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
```

## 🚀 Getting Started
Follow these steps to set up and run the project locally on your system.

1. Clone the Repository

```git clone [https://github.com/sumairanooruddin786-droid/Titanic-Survival-Predictor.git](https://github.com/sumairanooruddin786-droid/Titanic-Survival-Predictor.git)```
```cd Titanic-Survival-Predictor```

2. Install Required Dependencies
Make sure you have Python installed. Install the required libraries using ```pip```:
```pip install fastapi uvicorn pandas scikit-learn joblib jinja2 python-multipart```

## 📊 Model Training
To retrain or generate the ```titanic_model.pkl``` file, run the model training script:
```python train_model.py``

This script will:

Handle missing values for ```Age```, ```Fare```, and ```Embarked```.

Convert categorical features (```Sex```, ```Embarked```) into numerical format.

Train a RandomForestClassifier.

Save the trained model binary as ```titanic_model.pkl```.

## 🖥️ Running the Web Application
To launch the FastAPI web application, run Uvicorn from the root directory:
```uvicorn main:app --reload```

Once running, open your web browser and go to:
```👉 http://127.0.0.1:8000```

## 🔗 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Renders the main user interface form (`index.html`). |
| `/predict` | `POST` | Processes form input data and returns survival prediction with confidence score. |
