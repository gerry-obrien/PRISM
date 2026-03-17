from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import math

from app.services.listings import get_listings

router = APIRouter()


# CRITICAL: /listings/stats/summary must be registered before /listings/{listing_id}

@router.get("/listings/stats/summary")
def get_stats_summary():
    df = get_listings()
    valuation_counts = df["valuation"].value_counts().to_dict()
    return {
        "total": len(df),
        "mean_asking_price_eur": round(float(df["price_eur"].mean()), 2),
        "mean_estimated_price_eur": round(float(df["estimated_price_eur"].mean()), 2),
        "valuation_counts": {
            "Undervalued": int(valuation_counts.get("Undervalued", 0)),
            "Fair": int(valuation_counts.get("Fair", 0)),
            "Overvalued": int(valuation_counts.get("Overvalued", 0)),
        },
    }


@router.get("/listings/{listing_id}")
def get_listing(listing_id: str):
    df = get_listings()
    row = df[df["listing_id"] == listing_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Listing '{listing_id}' not found.")
    return row.iloc[0].to_dict()


@router.get("/listings")
def list_listings(
    arrondissement: Optional[int] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    dpe_rating: Optional[str] = Query(None),
    valuation: Optional[str] = Query(None),
    sort_by: str = Query("price_eur"),
    sort_order: str = Query("asc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
):
    df = get_listings().copy()

    if arrondissement is not None:
        df = df[df["arrondissement"] == arrondissement]
    if min_price is not None:
        df = df[df["price_eur"] >= min_price]
    if max_price is not None:
        df = df[df["price_eur"] <= max_price]
    if dpe_rating is not None:
        df = df[df["dpe_rating"] == dpe_rating]
    if valuation is not None:
        df = df[df["valuation"] == valuation]

    ascending = sort_order.lower() != "desc"
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    total = len(df)
    total_pages = max(1, math.ceil(total / page_size))
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "results": page_df.to_dict(orient="records"),
    }
