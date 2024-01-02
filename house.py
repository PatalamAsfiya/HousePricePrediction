import numpy as np
import pickle
import streamlit as st

# Define global dictionaries for mappings
location_mapping = {
    "Poranki": 8,
    "Kankipadu": 5,
    "Benz Circle": 0,
    "Gannavaram": 2,
    "Rajarajeswari Peta": 9,
    "Gunadala": 4,
    "Gollapudi": 3,
    "Enikepadu": 1,
    "Vidhyadharpuram": 10,
    "Penamaluru": 7,
    "Payakapuram": 6
}

status_mapping = {
    "Resale": 2,
    "Under Construction": 3,
    "Ready to move": 1,
    "New": 0
}

direction_mapping = {
    "Not Mentioned": 0,
    "East": 1,
    "West": 3,
    "NorthEast": 2
}

property_type_mapping = {
    "Apartment": 0,
    "Independent Floor": 1,
    "Independent House": 2,
    "Residential Plot": 3
}

with open('Model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict(bed, bath, loc, size, status, face, Type):
    selected_location_numeric = location_mapping[loc]
    selected_status_numeric = status_mapping[status]
    selected_direction_numeric = direction_mapping[face]
    selected_property_type_numeric = property_type_mapping[Type]

    input_data = np.array([[bed, bath, selected_location_numeric, size, selected_status_numeric, selected_direction_numeric, selected_property_type_numeric]])

    input_df = scaler.transform(input_data)

    return model.predict(input_df)[0]

if __name__ == '__main__':
    st.header('House Price Prediction')

    # Create a column layout to add the image alongside the prediction
    col1, col2 = st.columns([2, 1])

    bed = col1.slider('No of Bedrooms', max_value=10, min_value=1, value=2)
    bath = col1.slider('No of Bathrooms', max_value=7, min_value=1, value=2)
    loc = col1.selectbox("Select a Location", list(location_mapping.keys()))
    size = col1.number_input('Enter the Sq Feet', max_value=10000, min_value=100, value=1000, step=500)
    status = col1.selectbox("Select a Status", list(status_mapping.keys()))
    face = col1.selectbox("Select a Direction", list(direction_mapping.keys()))
    Type = col1.selectbox("Select a Property Type", list(property_type_mapping.keys()))

    result = predict(bed, bath, loc, size, status, face, Type)

    # Add an image to the second column (you need to specify the image URL)
    col2.image('https://img.freepik.com/free-photo/blue-house-with-blue-roof-sky-background_1340-25953.jpg', use_column_width=True)
    
    # Display the predicted value in the first column
    col2.write(f"The predicted value is: {result} Lakhs")