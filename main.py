import uvicorn
from fastapi import FastAPI

from inference import PredictionRequest, PredictionResponse, load_model, predict_email
from settings import settings

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.on_event("startup")
async def startup() -> None:
    load_model()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    return predict_email(request.email_text)


def main():
    uvicorn.run("main:app", host=settings.uvicorn_host, port=settings.uvicorn_port, reload=False)


if __name__ == "__main__":
    main()
