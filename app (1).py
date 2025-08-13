import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# --- Load Pre-computed Assets ---
# This function is cached to avoid reloading data on every interaction
@st.cache_data
def load_assets():
    """Loads all necessary data, models, and feature lists."""
    try:
        df = pd.read_csv("Cleaned_merged_suffixed.csv")
        with open('saved_features/all_crop_targets.json', 'r') as f:
            all_crops = json.load(f)
        return df, all_crops
    except FileNotFoundError:
        st.error("Critical files not found. Please run the `train_models.py` script first.")
        return None, None

df, all_crops = load_assets()

# --- App UI ---
if df is not None:
    st.title("🌾 Crop Yield Prediction Engine")
    st.markdown("Select a crop and input the relevant features to get a yield prediction from multiple machine learning models.")

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Prediction Inputs")
    
    # 1. Crop Selection
    selected_crop = st.sidebar.selectbox("Select Crop to Predict", all_crops)

    # 2. Load relevant features for the selected crop
    try:
        with open(f'saved_features/{selected_crop}_features.json', 'r') as f:
            feature_info = json.load(f)
        features = feature_info['features']
        state_columns = feature_info['state_columns']
        
        # Exclude state columns from the user input section
        input_features = [f for f in features if not f.startswith('State_')]
        
    except FileNotFoundError:
        st.error(f"Feature information for '{selected_crop}' not found. The model for this crop might not have been trained due to insufficient data.")
        st.stop()
        
    # 3. State Selection
    all_states = sorted([s.replace('State_', '') for s in state_columns])
    selected_state = st.sidebar.selectbox("Select State", all_states)

    # 4. Dynamic Feature Inputs
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Enter Feature Values:**")
    
    input_data = {}
    for feature in input_features:
        # Use a number input for each feature
        input_data[feature] = st.sidebar.number_input(feature, value=0.0, format="%.2f")

    # --- Prediction Logic ---
    if st.sidebar.button("Predict Yield", type="primary"):
        # 1. Create the feature DataFrame for prediction
        prediction_df = pd.DataFrame([input_data])
        
        # 2. Add the one-hot encoded state columns
        for state_col in state_columns:
            prediction_df[state_col] = 0
        
        # Set the selected state's column to 1
        selected_state_col_name = f'State_{selected_state}'
        if selected_state_col_name in prediction_df.columns:
            prediction_df[selected_state_col_name] = 1
            
        # 3. Ensure column order matches the training order
        prediction_df = prediction_df[features]

        # 4. Load Models and Make Predictions
        st.subheader(f"Predicted Yield for {selected_crop.replace('_', ' ')}")
        
        model_names = ['Linear Regression', 'Decision Tree', 'Random Forest', 'Ensemble (Voting)', 'Neural Network']
        
        cols = st.columns(len(model_names))
        
        for i, model_name in enumerate(model_names):
            try:
                # Load the pre-trained model
                model = joblib.load(f'saved_models/{selected_crop}_{model_name}.joblib')
                
                # Make a prediction
                prediction = model.predict(prediction_df)
                
                # Display the prediction in its own column
                with cols[i]:
                    st.metric(label=model_name, value=f"{prediction[0]:.2f} t/Ha")
            
            except FileNotFoundError:
                 with cols[i]:
                    st.warning(f"{model_name} model not found.")
