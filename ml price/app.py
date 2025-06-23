import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load data from uploaded file
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Train the model
@st.cache_resource
def train_model(data):
    X = data[['sqfeet', 'area_type']]
    y = data['price']

    # One-hot encode the 'area_type' column
    encoder = OneHotEncoder(sparse_output=False)
    area_encoded = encoder.fit_transform(X[['area_type']])
    area_encoded_df = pd.DataFrame(area_encoded, columns=encoder.get_feature_names_out(['area_type']))

    X_encoded = pd.concat([X[['sqfeet']].reset_index(drop=True), area_encoded_df.reset_index(drop=True)], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return model, encoder, mae

# Streamlit app
def main():
    st.title("🏠 House Price Predictor")
    st.write("Upload your housing dataset and predict house prices based on square footage and area type.")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            data = load_data(uploaded_file)
            st.success("✅ Data loaded successfully!")

            st.write("### Sample Data:")
            st.dataframe(data.head())

            # Train model
            model, encoder, mae = train_model(data)
            st.success(f"✅ Model trained successfully! Mean Absolute Error: {mae:.2f}")

            # Inputs
            sqfeet = st.number_input("Enter square footage:", min_value=0, value=1500, step=100)
            area_type = st.selectbox("Select area type:", options=data['area_type'].unique())

            if st.button("Predict Price"):
                input_data = pd.DataFrame({'sqfeet': [sqfeet], 'area_type': [area_type]})
                area_encoded = encoder.transform(input_data[['area_type']])
                area_encoded_df = pd.DataFrame(area_encoded, columns=encoder.get_feature_names_out(['area_type']))
                input_encoded = pd.concat([input_data[['sqfeet']], area_encoded_df], axis=1)

                predicted_price = model.predict(input_encoded)[0]
                st.success(f"💰 Predicted House Price: ${predicted_price:.2f}")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("📂 Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
