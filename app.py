import streamlit as st
import pandas as pd
import joblib
import math

# --------------------- Page Config ---------------------
st.set_page_config(
    page_title="Rail Fare Predictor",
    page_icon="🚂",
    layout="centered"
)

# --------------------- Load Model ---------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except:
        st.error("❌ Model file not found! Please make sure 'model.pkl' is in the same directory.")
        return None


# --------------------- Prediction Function ---------------------
def predict_fare(distance, duration, class_code, model):
    class_hierarchy = ['1A', '2A', '3A', 'SL', 'CC', '2S']
    class_mapping = {class_name: i for i, class_name in enumerate(class_hierarchy, 1)}
    luxury_mapping = {'1A': 6, '2A': 5, '3A': 4, 'CC': 2, '2S': 1, 'SL': 3}

    class_encoded = class_mapping[class_code]
    luxury_score = luxury_mapping[class_code]

    # Safe speed calculation
    speed_kmh = distance / (math.ceil(duration / 60)) if duration > 0 else 0
    if speed_kmh > 130:
        speed_kmh = 75  # cap unrealistic speeds

    km_per_minute = distance / duration if duration > 0 else 0
    speed_premium = speed_kmh * luxury_score
    efficiency_score = (distance / duration) * class_encoded if duration > 0 else 0
    is_express = 1 if speed_kmh > 60 else 0
    journey_intensity = distance / duration if duration > 0 else 0

    is_short_journey = 1 if distance <= 200 else 0
    is_medium_journey = 1 if 200 < distance <= 500 else 0
    is_long_journey = 1 if distance > 500 else 0

    distance_class = distance * class_encoded
    distance_luxury = distance * luxury_score
    speed_class = speed_kmh * class_encoded
    speed_luxury = speed_kmh * luxury_score
    distance_duration_ratio = distance / duration if duration > 0 else 0
    distance_duration_class = distance * duration * class_encoded
    express_premium = is_express * luxury_score

    distance_squared = distance ** 2
    duration_squared = duration ** 2
    speed_squared = speed_kmh ** 2

    # Binning placeholders
    distance_bin, duration_bin, speed_bin = 0, 0, 0

    input_data = [[
        distance, duration, class_encoded,
        luxury_score,
        speed_kmh, km_per_minute, speed_premium, efficiency_score,
        is_express, journey_intensity,
        is_short_journey, is_medium_journey, is_long_journey,
        distance_class, distance_luxury, speed_class, speed_luxury,
        distance_duration_ratio, distance_duration_class, express_premium,
        distance_squared, duration_squared, speed_squared,
        distance_bin, duration_bin, speed_bin
    ]]

    predicted_fare = model.predict(input_data)[0]
    return predicted_fare, speed_kmh


# --------------------- Streamlit App ---------------------
def main():
    st.title("🚂 Indian Railway Fare Predictor")
    # st.markdown("### Predict train fares using Machine Learning")
    st.markdown("---")

    model = load_model()
    if model is None:
        return

    st.header("📝 Journey Details")

    # ---- Distance ----
    distance = st.number_input(
        "Distance (km)",
        min_value=1,
        max_value=5000,
        value=200,
        help="Enter the journey distance in kilometers"
    )

    # ---- Duration (hours & minutes) ----
    st.markdown("### ⏱️ Journey Duration")
    col1, col2 = st.columns(2)
    with col1:
        hours = st.number_input("Hours", min_value=0, max_value=120, value=3)
    with col2:
        minutes = st.number_input("Minutes", min_value=0, max_value=59, value=0)
    duration = (hours * 60) + minutes

    # ---- Class ----
    class_code = st.selectbox(
        "Travel Class",
        options=['1A', '2A', '3A', 'CC', '2S', 'SL'],
        index=1,
        help="Select your preferred travel class"
    )

    class_descriptions = {
        '1A': 'First AC - Most luxurious',
        '2A': 'Second AC - Comfortable with AC',
        '3A': 'Third AC - Budget AC travel',
        'CC': 'Chair Car - Comfortable seating',
        '2S': 'Second Sitting - Basic seating',
        'SL': 'Sleeper Class - Most economical'
    }

    st.info(f"**{class_code}**: {class_descriptions[class_code]}")

    predict_btn = st.button("🚀 Predict Fare", type="primary")

    # ---- Prediction ----
    if predict_btn:
        with st.spinner('Calculating fare...'):
            fare, speed = predict_fare(distance, duration, class_code, model)

            # Warn if unrealistic
            if distance / (duration / 60) > 130 or speed < 20:
                st.warning("⚠️ Unrealistic input values. Check distance or duration inputs.")
            else:
                st.success("✅ Fare prediction completed!")

                # --- Two-Row Result Display ---
                st.markdown("### 💰 Predicted Fare")
                st.markdown(
                    f"<h2 style='text-align:center; color:#2ecc71;'>₹{fare:,.2f}</h2>",
                    unsafe_allow_html=True
                )

                st.markdown("---")

                st.markdown("### 🚄 Average Speed")
                st.markdown(
                    f"<h2 style='text-align:center; color:#3498db;'>{speed:.1f} km/h</h2>",
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    main()
