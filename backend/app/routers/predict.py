from fastapi import APIRouter

from app.models.schemas import PredictRequest, PredictResponse, MetricsResponse
from app.services.model import predict as model_predict, get_metrics

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    data = request.model_dump()
    estimated = model_predict(data)
    return PredictResponse(estimated_price_eur=estimated)


@router.get("/predict/metrics", response_model=MetricsResponse)
def metrics():
    m = get_metrics()
    return MetricsResponse(**m)
