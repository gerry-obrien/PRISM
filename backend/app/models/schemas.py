from typing import Literal
from pydantic import BaseModel


class PredictRequest(BaseModel):
    area_sqm: float
    num_rooms: int
    arrondissement: int
    floor: int
    has_elevator: bool
    is_new_build: bool
    year: int
    property_type: Literal["Apartment", "House"]
    dpe_rating: Literal["A", "B", "C", "D", "E", "F", "G"]
    building_condition: Literal["Good", "Average", "Poor"]


class PredictResponse(BaseModel):
    estimated_price_eur: float


class MetricsResponse(BaseModel):
    r2: float
    mae: float
    mape: float
