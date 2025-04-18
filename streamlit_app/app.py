import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from io import BytesIO

# --- Page configuration (MUST be the first Streamlit command) ---
st.set_page_config(page_title="Istanbul Airbnb Price Estimator", page_icon="🏡")

# --- App Title ---
st.markdown("<h1 style='text-align: center;'>🏡 Istanbul Airbnb Price Estimator</h1>", unsafe_allow_html=True)
st.markdown("This app predicts the **nightly price** of an Airbnb listing in Istanbul using a machine learning model trained on real Airbnb data.")

# --- Load model and feature names ---
@st.cache_resource
def load_model():
    url = "https://huggingface.co/mnalbantli/airbnb-istanbul-model/resolve/main/rf_airbnb_model.pkl"
    response = requests.get(url)
    model, feature_names = joblib.load(BytesIO(response.content))
    return model, feature_names

model, feature_names = load_model()

# --- User Inputs ---
room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
review_score = st.slider("Review Score", 1.0, 5.0, 4.5, step=0.01)
min_nights = st.number_input("Minimum Nights", min_value=1, value=3)
superhost = st.checkbox("Superhost?", value=True)
instant = st.checkbox("Instant Bookable?", value=True)

districts = sorted([
    'Arnavutkoy', 'Atasehir', 'Avcilar', 'Bagcilar', 'Bahcelievler', 'Bakirkoy',
    'Basaksehir', 'Bayrampasa', 'Besiktas', 'Beykoz', 'Beyoglu', 'Buyukcekmece',
    'Catalca', 'Cekmekoy', 'Esenler', 'Esenyurt', 'Eyupsultan', 'Fatih',
    'Gaziosmanpasa', 'Gungoren', 'Kadikoy', 'Kagithane', 'Kartal', 'Kucukcekmece',
    'Maltepe', 'Pendik', 'Sancaktepe', 'Sariyer', 'Silivri', 'Sisli',
    'Sultanbeyli', 'Sultangazi', 'Tuzla', 'Umraniye', 'Uskudar', 'Zeytinburnu'
])
district = st.selectbox("District", districts)

# --- Build feature vector dynamically ---
X = pd.DataFrame(columns=feature_names)
X.loc[0] = 0  # Initialize all features to zero

# Only fill values that exist in model
if f"room_type_{room_type}" in feature_names:
    X.at[0, f"room_type_{room_type}"] = 1
if f"neighbourhood_cleansed_{district}" in feature_names:
    X.at[0, f"neighbourhood_cleansed_{district}"] = 1
if "minimum_nights" in feature_names:
    X.at[0, "minimum_nights"] = min_nights
if "review_scores_rating" in feature_names:
    X.at[0, "review_scores_rating"] = review_score
if "instant_bookable_t" in feature_names:
    X.at[0, "instant_bookable_t"] = int(instant)
if "host_is_superhost_t" in feature_names:
    X.at[0, "host_is_superhost_t"] = int(superhost)

# --- Predict Button ---
if st.button("💰 Show Estimated Price"):
    try:
        predicted_price = model.predict(X)[0]
        st.success(f"Estimated Nightly Price: **{round(predicted_price, 2)} TRY**")
    except Exception as e:
        st.error(f"Prediction failed. Details: {e}")

# --- Footer / References ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 14px;'>
    🔗 Model hosted on <a href='https://huggingface.co/mnalbantli/airbnb-istanbul-model' target='_blank'>Hugging Face</a> |
    📊 Data from <a href='https://insideairbnb.com/get-the-data.html' target='_blank'>Inside Airbnb</a> |
    👨‍💻 Built by Mustafa using <a href='https://streamlit.io/' target='_blank'>Streamlit</a> + <a href='https://scikit-learn.org/' target='_blank'>scikit-learn</a>
    <br><br>
    🔗 <a href='https://www.linkedin.com/in/mustafanalbantli/' target='_blank'>LinkedIn</a> |
    💻 <a href='https://github.com/mnalbantli' target='_blank'>GitHub</a>
</div>
""", unsafe_allow_html=True)

