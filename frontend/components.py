#placeholder

import streamlit as st
import pandas as pd

#sidebar filter selection

rating_order = {
    "A": 7,
    "B": 6,
    "C": 5,
    "D": 4,
    "E": 3,
    "F": 2,
    "G": 1,
    }

def filter_selection(df):
    with st.sidebar:
        with st.form("filters"):
            arrondissement_options = ["1","2","3","4","5","6","7","8","9","10",
                                      "11","12","13","14","15","16","17","18","19"
                                      "20"]

            selected_arrondissements = st.multiselect(
                "Arrondissement",
                options=arrondissement_options
            )

            min_price = st.number_input(
                "Minimum price",
                min_value=0,
                value=0,
                step=50000
            )

            max_price = st.number_input(
                "Maximum price",
                min_value=0,
                value=10000000,
                step=50000
            )



            min_rooms = st.slider(
                "Minimum number of rooms",
                min_value=1,
                max_value = 10,
                value=1,
                step=1
            )

            min_sqm = st.number_input(
                "Minimum area (square meters)",
                min_value=0,
                value=0,
                step=5
            )

            min_energy_label = st.select_slider(
                "Minimum energy rating",
                options=["G", "F", "E", "D", "C", "B", "A"],
                value="G"
            )

            submitted = st.form_submit_button("Apply filters")

    return {
        "selected_arrondissements": selected_arrondissements,
        "min_price": min_price,
        "max_price": max_price,
        "min_rooms": min_rooms,
        "min_sqm": min_sqm,
        "min_energy_rating": min_energy_label,
        "submitted": submitted,
    }

#function that filters df based on user sidebar input

def apply_filters(df, filters):
    filtered = df.copy()

    if filters["selected_arrondissements"]:
        filtered = filtered[filtered["arrondissement"].astype('str').isin(filters["selected_arrondissements"])]
    filtered = filtered[filtered["price_eur"] >= filters["min_price"]]
    filtered = filtered[filtered["price_eur"] <= filters["max_price"]]
    filtered = filtered[filtered["num_rooms"] >= filters["min_rooms"]]
    filtered = filtered[filtered["area_sqm"] >= filters["min_sqm"]]
    filtered = filtered[filtered["dpe_rating"].map(rating_order)>= rating_order[filters["min_energy_rating"]]
    ]

    return filtered

