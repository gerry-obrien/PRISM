
#import required functions
import streamlit as st
from pathlib import Path
import pandas as pd
from data import load_data
from components import filter_selection, apply_filters


st.title("Property Explorer")

#load the data that is generated in data folder
df = load_data()

#prompts user and stores filter responses
filters = filter_selection(df)


#applies filtering function to refine results to user input
filtered_df = apply_filters(df, filters)

#displays filtered dataframe
st.subheader("Results")
st.write(f"{len(filtered_df)} properties found")
st.dataframe(filtered_df)

#displays filtered map

st.map(filtered_df)
