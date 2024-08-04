import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

image_path = 'Inno_logo_.png'  # Replace with your actual PNG image file path

# Specify the desired width and height
st.image(image_path, width=400)

# Load the pre-trained model (which includes the scaler)
with open("best_model_lr.pkl", "rb") as model_file:
    pipeline = pickle.load(model_file)

# Define the Streamlit app
st.title("Anemia Classification")

# Input fields for user to provide feature values
sex = st.selectbox('Sex', ['M', 'F'])
red_pixel = st.number_input('%Red Pixel', min_value=0.0, max_value=100.0, step=0.01)
green_pixel = st.number_input('%Green pixel', min_value=0.0, max_value=100.0, step=0.01)
blue_pixel = st.number_input('%Blue pixel', min_value=0.0, max_value=100.0, step=0.01)
hb = st.number_input('Hb', min_value=0.0, step=0.01)

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    'Sex': [sex],
    '%Red Pixel': [red_pixel],
    '%Green pixel': [green_pixel],
    '%Blue pixel': [blue_pixel],
    'Hb': [hb]
})

# Encode categorical variables
input_data['Sex'] = input_data['Sex'].map({'M': 1, 'F': 0})

# Predict with the model
if st.button('Classify'):
    prediction = pipeline.predict(input_data)
    result = "Anemic" if prediction[0] == 1 else "Not Anemic"
    st.write(f"The model predicts: {result}")

    # Show an image based on the prediction
    if result == "Anemic":
        st.image(r"Anemia_Yes.png", caption="Anemia Detected")
    else:
        st.image(r"Anemia-Normal.png", caption="No Anemia Detected")
