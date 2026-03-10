from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model_filename: str = "logistic_regression_model.pkl"
    models_dir: Path = Path("saved_models")
    scaler_filename: str = "feature_scaler.pkl"
    uvicorn_host: str = "0.0.0.0"
    uvicorn_port: int = 8001


settings = Settings()
