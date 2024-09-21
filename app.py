import streamlit as st
import pandas as pd
import pickle

# Load the model and scaler
with open('best_linear_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Set page configuration for better look
st.set_page_config(
    page_title="Music Popularity Predictor",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded",
)

# App Title and Description
st.title('🎶 Predict The Popularity Of Songs Based on Parameters')

st.write("""
    This app predicts the popularity of a track based on its features:
    - **Duration**: Length of the track (in seconds).
    - **Danceability**: How suitable the track is for dancing (0.0 to 1.0).
    - **Energy**: Intensity and activity of the track (0.0 to 1.0).
    - **Instrumentalness**: Instrumental nature of the track (0.0 to 1.0).
    - **Liveness**: Likelihood of a live performance (0.0 to 1.0).
""")

# Organizing layout into two columns
col1, col2 = st.columns(2)

with col1:
    # Sliders for numerical input
    duration = st.slider('Duration (seconds)', min_value=0.0, max_value=600.0, value=200.0)
    danceability = st.slider('Danceability', min_value=0.0, max_value=1.0, step=0.01, value=0.5)

with col2:
    energy = st.slider('Energy', min_value=0.0, max_value=1.0, step=0.01, value=0.7)
    instrumentalness = st.slider('Instrumentalness', min_value=0.0, max_value=1.0, step=0.01, value=0.1)
    liveness = st.slider('Liveness', min_value=0.0, max_value=1.0, step=0.01, value=0.3)

# Predict button
if st.button('Predict Popularity'):
    # Prepare input data
    input_data = pd.DataFrame([[duration, danceability, energy, instrumentalness, liveness]], 
                              columns=['Duration', 'Danceability', 'Energy', 'Instrumentalness', 'Liveness'])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    
    # Display prediction
    st.write(f'🎯 **Predicted Popularity (1 to 100)**: {prediction[0]:.2f}')
    
    # Display interpretation of the score
    if prediction > 57:
        st.success("🎉 The music track might be **Excellent**! Get ready for a hit!")
    elif prediction > 51:
        st.info("👍 The music track might be **Good**! It has potential!")
    elif prediction > 46:
        st.warning("🙂 The music track might be **Average**. Could work with the right audience.")
    elif prediction > 31:
        st.warning("😕 The music track might be **Below Average**. Might need more work!")
    else:
        st.error("😢 The music track might be **Poor**. Not the best track.")
