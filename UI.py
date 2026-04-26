# app.py

import streamlit as st
from model import calculate_score, update_weights
import pandas as pd

df = pd.read_csv("users.csv")

users = df.to_dict(orient="records")

# SESSION STATE INIT 
if "weights" not in st.session_state:
    st.session_state.weights = {'w1': 0.5, 'w2': 0.3, 'w3': 0.2}

if "results" not in st.session_state:
    st.session_state.results = []

if "last_action" not in st.session_state:
    st.session_state.last_action = None

st.title(" Intelligent Hybrid Recommendation System :)")

# USER INPUT 
bio = st.text_area("Enter your bio")
mbti = st.selectbox("Select MBTI", ["INTJ", "ENFP", "INFJ", "ENTP"])
location = st.selectbox("Select Location", ["Delhi", "Mumbai", "Bangalore"])

current_user = {"bio": bio, "mbti": mbti, "location": location}

# FIND MATCHES 
if st.button("Find Matches"):
    results = []

    for user in users:
        score, features = calculate_score(current_user, user, st.session_state.weights)
        results.append((user, score, features))

    results = sorted(results, key=lambda x: x[1], reverse=True)

    st.session_state.results = results

# DISPLAY ONLY TOP 3 
if st.session_state.results:
    st.subheader(" Top 5 Matches")

    for i, (user, score, features) in enumerate(st.session_state.results[:5]):
        st.write(f"User {user['id']} → Score: {round(score * 100, 2)}%")
        st.write(f"Bio: {user['bio']}")
        st.write(f"MBTI: {user['mbti']} | Location: {user['location']}")

        col1, col2 = st.columns(2)

        if col1.button(f" Accept {user['id']}", key=f"accept_{user['id']}"):
            st.session_state.weights = update_weights(
                st.session_state.weights, 1, features
            )
            st.session_state.last_action = f"Accepted User {user['id']}"

        if col2.button(f" Reject {user['id']}", key=f"reject_{user['id']}"):
            st.session_state.weights = update_weights(
                st.session_state.weights, 0, features
            )
            st.session_state.last_action = f"Rejected User {user['id']}"

        st.markdown("---")

# SHOW LEARNING 
st.subheader(" System Learning")

if st.session_state.last_action:
    st.success(f"Last Action: {st.session_state.last_action}")

st.write("### Updated Weights:")
st.write(st.session_state.weights)