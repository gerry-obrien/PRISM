
#import required functions
import streamlit as st
from pathlib import Path
import pandas as pd
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
