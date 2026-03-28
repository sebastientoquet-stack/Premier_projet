from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import os
import pandas as pd
import numpy as np


app = FastAPI(
    title="API de Classification Automatique",
    description="API pour classifier automatiquement des informations (Projet 4)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class TextInput(BaseModel):
    text: str = Field(..., description="Texte à classifier", min_length=1)
    model_version: Optional[str] = Field("default", description="Version du modèle à utiliser")

class PredictionOutput(BaseModel):
    text: str
    prediction: int
    prediction_label: str
    confidence: float
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str

model = None
vectorizer = None

LABELS = {
    0: "Classe 0",
    1: "Classe 1",
    2: "Classe 2"
}

def load_model():
    global model, vectorizer
    model_path = "models/model.pkl"
    vectorizer_path = "models/vectorizer.pkl"
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Modèle chargé depuis {model_path}")
    else:
        print(f"Modèle non trouvé: {model_path}")
        print("L'API fonctionnera en mode démo")
    
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
        print(f"Vectorizer chargé depuis {vectorizer_path}")

def preprocess_text(text: str) -> str:
    text = text.lower().strip()
    return text

@app.on_event("startup")
async def startup_event():
    load_model()
    print("API démarrée - Bienvenue sur l'API de classification!")

@app.get("/", response_model=dict)
def root():
    return {
        "message": "API de classification - Projet 4",
        "status": "online",
        "endpoints": ["/health", "/predict", "/docs"]
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version="1.0.0"
    )

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: TextInput):
    try:
        processed_text = preprocess_text(input_data.text)
        
        if model is not None and vectorizer is not None:
            text_vectorized = vectorizer.transform([processed_text])
            prediction = model.predict(text_vectorized)[0]
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = float(max(probabilities))
        else:
            prediction = 0
            confidence = 0.95
        
        prediction_label = LABELS.get(prediction, f"Classe {prediction}")
        
        return PredictionOutput(
            text=input_data.text,
            prediction=int(prediction),
            prediction_label=prediction_label,
            confidence=confidence,
            model_version="1.0.0"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
