import logging
from collections.abc import Mapping

from fastapi import FastAPI, File, HTTPException, UploadFile

from functions.model import get_model
from functions.preprocessing import preprocess_image
from functions.schema import PredictionResponse

logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/health")
async def health(check_model: bool = False):
    if not check_model:
        return {"status": "ok"}

    try:
        _ = get_model("milho")
        return {"status": "ok", "model": "loaded"}
    except Exception as exc:
        logger.exception("Healthcheck falhou ao carregar modelo.")
        raise HTTPException(status_code=503, detail=f"model_unavailable: {exc}") from exc


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        image = preprocess_image(file.file)
        image_np = image.numpy()
    except Exception as exc:
        logger.exception("Falha ao processar imagem enviada.")
        raise HTTPException(status_code=400, detail=f"invalid_image: {exc}") from exc

    try:
        model = get_model("milho")
        prediction = model.predict(image_np)
        payload = prediction[0]
    except Exception as exc:
        logger.exception("Falha na predição.")
        raise HTTPException(status_code=503, detail=f"prediction_unavailable: {exc}") from exc

    if not isinstance(payload, Mapping):
        logger.error("Formato inesperado da predição: %s", type(payload).__name__)
        raise HTTPException(
            status_code=500,
            detail="invalid_prediction_payload: esperado objeto com campos 'prediction' e 'confidence'",
        )

    if "prediction" not in payload or "confidence" not in payload:
        logger.error("Chaves ausentes na predição: %s", payload)
        raise HTTPException(
            status_code=500,
            detail="invalid_prediction_payload: faltam campos obrigatórios",
        )

    return PredictionResponse(
        prediction=str(payload["prediction"]),
        confidence=float(payload["confidence"]),
    )
