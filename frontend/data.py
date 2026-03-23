#placeholder
import pandas as pd
import streamlit as st
from pathlib import Path

#function that loads data from generated listings.csv file and cleans data
def load_data():
    #loading data
    BASE_DIR = Path(__file__).resolve().parent
    csv_path = BASE_DIR.parent / "data" / "listings.csv"
    df = pd.read_csv(csv_path)
    #cleaning data - making necessary columns numeric
    numeric_cols = [
        "price_eur",
        "area_sqm",
        "num_rooms",
        "year",
        "latitude",
        "longitude",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    return df
    
