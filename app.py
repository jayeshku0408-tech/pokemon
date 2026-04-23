import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt



# Load models
model_leg = joblib.load("legendary_model.pkl")
model_type = joblib.load("type_model.pkl")
type_encoder = joblib.load("type_encoder.pkl")

st.set_page_config(page_title="Pokemon Predictor", layout="wide")

# Title
st.title("🔥 Pokemon Predictor Pro")
st.markdown("Predict **Legendary Status** and **Type** with AI")

# Sidebar
st.sidebar.header("🎛️ Customize Stats")

hp = st.sidebar.slider("HP", 1, 255, 50)
attack = st.sidebar.slider("Attack", 1, 200, 50)
defense = st.sidebar.slider("Defense", 1, 250, 50)
sp_atk = st.sidebar.slider("Sp. Atk", 1, 200, 50)
sp_def = st.sidebar.slider("Sp. Def", 1, 250, 50)
speed = st.sidebar.slider("Speed", 1, 200, 50)
generation = st.sidebar.selectbox("Generation", [1, 2, 3, 4, 5, 6])

# Prepare input
features = np.array([[hp, attack, defense, sp_atk, sp_def, speed, generation]])

# Layout columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Stats Overview")
    
    stats_df = pd.DataFrame({
        "Stats": ["HP", "Attack", "Defense", "Sp.Atk", "Sp.Def", "Speed"],
        "Values": [hp, attack, defense, sp_atk, sp_def, speed]
    })
    
    st.bar_chart(stats_df.set_index("Stats"))

with col2:
    st.subheader("⚡ Total Power")
    total = hp + attack + defense + sp_atk + sp_def + speed
    st.metric(label="Total Stats", value=total)

# Prediction
if st.button("🔮 Predict"):
    
    leg_pred = model_leg.predict(features)[0]
    leg_prob = model_leg.predict_proba(features)[0][1]
    
    type_pred = model_type.predict(features)[0]
    type_name = type_encoder.inverse_transform([type_pred])[0]
    
    st.subheader("🎯 Results")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if leg_pred:
            st.success("🏆 Legendary Pokemon!")
        else:
            st.info("Not Legendary")
        
        st.write(f"Confidence: **{leg_prob:.2%}**")
    
    with col4:
        st.write(f"🎨 Predicted Type: **{type_name}**")
    
    # Fun insights
    st.subheader("🧠 Insights")
    
    if total > 600:
        st.success("💪 Extremely Strong Pokemon!")
    elif total > 400:
        st.warning("⚔️ متوسط Strength Pokemon")
    else:
        st.info("🐣 Low Stats Pokemon")
    
    if speed > 120:
        st.write("⚡ Very Fast!")
    if attack > sp_atk:
        st.write("🗡️ Physical Attacker")
    else:
        st.write("✨ Special Attacker")

# Footer warning
st.warning("⚠️ Type prediction accuracy is low (~22%). Results may not be reliable.")
