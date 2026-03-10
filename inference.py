from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field

from settings import settings
from vectorizer import email_to_vector


class PredictionRequest(BaseModel):
    email_text: str = Field(min_length=1, description="Raw email content to classify.")


class PredictionResponse(BaseModel):
    model_filename: str
    prediction: int
    label: str
    probability: float | None = None


def _resolve_artifact(filename: str) -> Path:
    artifact_path = settings.models_dir / filename
    if not artifact_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Artifact '{artifact_path.name}' was not found in '{settings.models_dir}'.",
        )
    return artifact_path


@lru_cache(maxsize=1)
def load_model() -> Any:
    return joblib.load(_resolve_artifact(settings.model_filename))


@lru_cache(maxsize=1)
def load_scaler() -> Any | None:
    scaler_path = settings.models_dir / settings.scaler_filename
    if not scaler_path.exists():
        return None
    return joblib.load(scaler_path)


def predict_email(email_text: str) -> PredictionResponse:
    features = np.array(email_to_vector(email_text), dtype=float).reshape(1, -1)

    scaler = load_scaler()
    if scaler is not None:
        features = scaler.transform(features)

    model = load_model()
    prediction = int(model.predict(features)[0])

    probability: float | None = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0][prediction])

    return PredictionResponse(
        model_filename=settings.model_filename,
        prediction=prediction,
        label="spam" if prediction == 1 else "ham",
        probability=probability,
    )
