from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI(title="Titanic Survival Prediction API")

# Trained model load karein
model = joblib.load("titanic_model.pkl")

# Templates directory link karein
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    pclass: int = Form(...),
    sex: int = Form(...),
    age: float = Form(...),
    sibsp: int = Form(...),
    parch: int = Form(...),
    fare: float = Form(...),
    embarked: int = Form(...)
):
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    if prediction == 1:
        result = "Survived"
        confidence = round(probabilities[1] * 100, 2)
    else:
        result = "Did Not Survive"
        confidence = round(probabilities[0] * 100, 2)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": result, 
            "confidence": confidence
        }
    )