import streamlit as st
from pathlib import Path
import pandas as pd
import requests
from data import load_data
from components import filter_selection, apply_filters
from authentication import init_db, create_user, authenticate_user

#--- authentication:

#initialize user db
init_db()


#creates variable that stores whether user is logged in / in database
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

#login prompt
@st.dialog("Login")

#use autheticate_user and create_user functions from authentication.py to prompt
#user logins and new account creations
def login_dialog():
    tab1, tab2 = st.tabs(["Log in", "Create account"])

    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Log in"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        username = st.text_input("New username", key="signup_username")
        password = st.text_input("New password", type="password", key="signup_password")

        if st.button("Create account"):
            if create_user(username, password):
                st.success("Account created. You can now log in.")
            else:
                st.error("Username already exists")


#stops if session not authenticated
if not st.session_state.authenticated:
    login_dialog()
    st.stop()

st.write(f"Logged in as: {st.session_state.username}")

#option to log out of authenitcated session
if st.button("Log out"):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()


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
