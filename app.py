import streamlit as st
import numpy as np
import pickle


import os

# Change working directory
#os.chdir(r'C:\Users\COMD\Desktop\AI-ML Project\Final Map\Crop_recommendation')


# Load the model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Define crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}
st.image("plants.jpg", use_column_width=True)

# App title
st.title("Crop Recommendation System")

# Input fields
st.header("Enter the soil and environmental parameters:")

N = st.number_input("Nitrogen Content (N)", min_value=0.0, step=0.1)
P = st.number_input("Phosphorus Content (P)", min_value=0.0, step=0.1)
K = st.number_input("Potassium Content (K)", min_value=0.0, step=0.1)
temp = st.number_input("Temperature (°C)", min_value=-10.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, step=0.1)
ph = st.number_input("Soil pH Level", min_value=0.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)

# Prediction logic
if st.button("Predict Best Crop"):
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        st.success(f"{crop} is the best crop to be cultivated in the given conditions.")
    else:
        st.error("Sorry, we could not determine the best crop to be cultivated with the provided data.")
