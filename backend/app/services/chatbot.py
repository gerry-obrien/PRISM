"""
Chatbot service — sends user questions to a local Ollama LLM
along with a summary of the listings data so it can answer
investment-related questions.
"""

import requests
import pandas as pd
from app.services.listings import get_listings


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:1b"


def _build_data_summary() -> str:
    """
    Build a short text summary of the listings data.
    We don't send the full CSV to the LLM — just the key stats
    so it has enough context to answer questions.
    """
    df = get_listings()

    total = len(df)
    avg_price = df["price_eur"].mean()
    avg_estimated = df["estimated_price_eur"].mean()
    avg_area = df["area_sqm"].mean()

    # Count undervalued / fair / overvalued
    val_counts = df["valuation"].value_counts().to_dict()
    undervalued = val_counts.get("Undervalued", 0)
    fair = val_counts.get("Fair", 0)
    overvalued = val_counts.get("Overvalued", 0)

    # Stats per arrondissement
    arr_stats = (
        df.groupby("arrondissement")
        .agg(
            count=("price_eur", "size"),
            avg_price=("price_eur", "mean"),
            avg_estimated=("estimated_price_eur", "mean"),
            avg_area=("area_sqm", "mean"),
        )
        .round(0)
    )

    arr_lines = []
    for arr, row in arr_stats.iterrows():
        arr_lines.append(
            f"  Arr. {int(arr)}: {int(row['count'])} listings, "
            f"avg asking {int(row['avg_price'])}€, "
            f"avg estimated {int(row['avg_estimated'])}€, "
            f"avg area {int(row['avg_area'])}m²"
        )

    # Top 5 best deals (most undervalued)
    if "price_delta_pct" in df.columns:
        best_deals = df.nsmallest(5, "price_delta_pct")
        deals_lines = []
        for _, row in best_deals.iterrows():
            deals_lines.append(
                f"  {row['listing_id']}: Arr. {int(row['arrondissement'])}, "
                f"{int(row['area_sqm'])}m², "
                f"asking {int(row['price_eur'])}€, "
                f"estimated {int(row['estimated_price_eur'])}€, "
                f"delta {row['price_delta_pct']:.1f}%"
            )
        deals_text = "\n".join(deals_lines)
    else:
        deals_text = "  No delta data available."

    summary = f"""PARIS REAL ESTATE MARKET DATA (PRISM)
Total listings: {total}
Average asking price: {int(avg_price)}€
Average ML-estimated price: {int(avg_estimated)}€
Average area: {int(avg_area)}m²
Valuation breakdown: {undervalued} undervalued, {fair} fair, {overvalued} overvalued

Stats by arrondissement:
{chr(10).join(arr_lines)}

Top 5 best deals (most undervalued):
{deals_text}
"""
    return summary


def _build_system_prompt(data_summary: str) -> str:
    """The system prompt tells the LLM who it is and what data it has."""
    return f"""You are PRISM Investment Advisor, a helpful assistant for people
looking to buy property in Paris. You have access to a dataset of 1,000
property listings with ML-estimated fair prices.

Here is the current market data:

{data_summary}

Rules:
- Answer in a helpful, concise way.
- When the user asks about specific arrondissements or price ranges, use the data above.
- If the user asks about investment returns, you can do simple calculations:
  assume ~3.5% gross rental yield in Paris on average, ~1.5% annual price growth,
  and typical charges/taxes of ~25% of rent. Adjust by arrondissement if relevant.
- If you don't know something, say so. Don't make up listings that aren't in the data.
- Keep answers short — 2 to 4 paragraphs max.
- You can speak French or English depending on what the user uses.
"""


def get_listings_for_query(arrondissement=None, valuation=None, limit=10):
    """
    Fetch specific listings to include in the context when the user
    asks about a specific area or valuation type.
    """
    df = get_listings()
    if arrondissement is not None:
        df = df[df["arrondissement"] == arrondissement]
    if valuation is not None:
        df = df[df["valuation"] == valuation]

    df = df.head(limit)

    if df.empty:
        return "No listings match these criteria."

    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"- {row['listing_id']}: {int(row['area_sqm'])}m², "
            f"{int(row['num_rooms'])} rooms, floor {int(row['floor'])}, "
            f"DPE {row['dpe_rating']}, "
            f"asking {int(row['price_eur'])}€, "
            f"estimated {int(row['estimated_price_eur'])}€ "
            f"({row['valuation']})"
        )
    return "\n".join(lines)


def chat(user_message: str, conversation_history: list) -> str:
    """
    Send a message to Ollama and get a response.

    conversation_history is a list of dicts like:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    data_summary = _build_data_summary()
    system_prompt = _build_system_prompt(data_summary)

    # Build the messages list for Ollama
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        return result["message"]["content"]

    except requests.ConnectionError:
        return (
            "Could not connect to Ollama. "
            "Make sure it's running (ollama serve) and the model is pulled (ollama pull mistral)."
        )
    except Exception as e:
        return f"Error calling the LLM: {str(e)}"
