import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Train the model
@st.cache_resource
def train_model(data):
    X = data[['sqfeet', 'area_type']]
    y = data['price']
    
    # One-hot encode the 'area_type' column
    encoder = OneHotEncoder(sparse_output=False)
    area_encoded = encoder.fit_transform(X[['area_type']])
    X_encoded = pd.concat(
        [X[['sqfeet']], pd.DataFrame(area_encoded, columns=encoder.get_feature_names_out())], axis=1
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, encoder, mae

# Streamlit app
def main():
    st.title("House Price Predictor")
    st.write("Enter the details of the house to predict its price.")
    
    # File path for data
    file_path = r"D:\practice files\ml price\real_estate_data.csv"  # Use raw string for Windows paths
    try:
        data = load_data(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure the file path is correct.")
        return
    
    st.write("Sample Data:")
    st.write(data.head())
    
    # Train the model
    model, encoder, mae = train_model(data)
    st.write(f"Model trained! Mean Absolute Error: {mae:.2f}")
    
    # User input
    sqfeet = st.number_input("Enter the square footage:", min_value=0, value=1500, step=100)
    area_type = st.selectbox("Select the type of area:", options=data['area_type'].unique())
    
    # Predict price
    if st.button("Predict Price"):
        # Prepare input
        input_data = pd.DataFrame({'sqfeet': [sqfeet], 'area_type': [area_type]})
        area_encoded = encoder.transform(input_data[['area_type']])
        input_encoded = pd.concat(
            [input_data[['sqfeet']], pd.DataFrame(area_encoded, columns=encoder.get_feature_names_out())], axis=1
        )
        
        # Predict
        predicted_price = model.predict(input_encoded)[0]
        st.write(f"The predicted price of the house is: ${predicted_price:.2f}")

if __name__ == "__main__":
    main()
