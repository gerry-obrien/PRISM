import streamlit as st
from pathlib import Path
import pandas as pd
import requests
from data import load_data
from components import filter_selection, apply_filters

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="PRISM", layout="wide")
st.title("PRISM - Paris Real Estate")

# Two tabs: the existing property explorer + the new chatbot
tab_explorer, tab_advisor = st.tabs(["Property Explorer", "Investment Advisor"])

# ─── Tab 1: Property Explorer (existing code, unchanged) ─────────────────────

with tab_explorer:
    df = load_data()
    filters = filter_selection(df)
    filtered_df = apply_filters(df, filters)

    st.subheader("Results")
    st.write(f"{len(filtered_df)} properties found")
    st.dataframe(filtered_df)
    st.map(filtered_df)

# ─── Tab 2: Investment Advisor Chatbot ────────────────────────────────────────

with tab_advisor:
    st.subheader("Investment Advisor")
    st.caption(
        "Ask questions about the Paris property market, "
        "find undervalued listings, or estimate investment returns."
    )

    # Keep chat history in session state so it persists across reruns
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask something about Paris real estate...")

    if user_input:
        # Show the user message right away
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Call our backend API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "message": user_input,
                            "history": st.session_state.chat_history[:-1],
                        },
                        timeout=120,
                    )
                    response.raise_for_status()
                    reply = response.json()["reply"]
                except requests.ConnectionError:
                    reply = (
                        "Could not reach the API. "
                        "Make sure the backend is running (uvicorn)."
                    )
                except Exception as e:
                    reply = f"Error: {str(e)}"

            st.write(reply)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
