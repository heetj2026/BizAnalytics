import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Crop Yield Predictor & Analyzer",
    page_icon="🌾",
    layout="wide"
)

# --- Custom CSS for Styling ---

st.markdown("""
<style>
    /* Main App background */
    .stApp {
        background-color: #053229; /* Light gray background */
        color: white; /* Dark text for readability */
    }
    /* Sidebar styling */
    .st-sidebar {
        background-color: #; /* White sidebar */
    }
    /* Title color */
    h1 {
        color: #FFEDD1; /* A darker, professional green */
    }
    /* Metric label and value */
    .stMetricLabel {
        color: white;
    }
    .stMetricValue {
        color: white;
    }
    /* Button styling */
    .stButton>button {
        background-color: #FFEDD1;
        color: black;
        border-radius: 8px;
        border: none;
    }hi
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
		gap: 2px;
	}
	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		background-color: #355E58; /* Lighter gray for inactive tabs */
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
        padding-left: 10px;
        padding-right: 10px;
        font-weight: 800;
        font-size: 30px;
    }
    .stTabs [aria-selected="true"] {
  		background-color: #053229;
        color: #FFEDD1;
	}
</style>
""", unsafe_allow_html=True)



def clean_name(name):
    """Cleans up raw column names for display, removing duplicates."""
    name = name.replace('_Yield', '')
    parts = name.split('_')
    unique_parts = list(dict.fromkeys(parts))
    return ' '.join(unique_parts).strip().title()


@st.cache_data
def load_assets():
    """Loads all necessary data for the app."""
    try:
        with open('saved_features/trained_crop_models.json', 'r') as f:
            trained_crops = json.load(f)
        cleaned_crops_map = {clean_name(crop): crop for crop in trained_crops}
        
        eda_df = pd.read_csv("All_data_cleaned.csv")
        
        # --- FIX: Pre-calculate average yields for the high/low indicator ---
        average_yields = {}
        for crop_name in trained_crops:
            if crop_name in eda_df.columns and pd.api.types.is_numeric_dtype(eda_df[crop_name]):
                # Calculate mean only on non-zero values
                non_zero_yields = eda_df[eda_df[crop_name] > 0][crop_name]
                if not non_zero_yields.empty:
                    average_yields[crop_name] = non_zero_yields.mean()
        
        return cleaned_crops_map, eda_df, average_yields
    except FileNotFoundError:
        st.error("Model or data files not found. Please run the `train_models.py` and data cleaning scripts first.")
        return None, None, None

cleaned_crops_map, eda_df, average_yields = load_assets()


if cleaned_crops_map and eda_df is not None and average_yields is not None:
    
    st.title("Indian Crop Yield Predictor & Analyzer")
    
    tab1, tab2, tab3 = st.tabs(["Predictor", "Exploratory Data Analysis (EDA)", "Explanation"])


    with tab1:
        # --- Sidebar for User Inputs ---
        st.sidebar.header("Prediction Inputs")
        
        selected_season = st.sidebar.radio(
            "Select Crop Season",
            ('Kharif', 'Rabi'),
            horizontal=True,
            key='season_radio'
        )

        filtered_crops = {
            clean: original for clean, original in cleaned_crops_map.items() 
            if selected_season.lower().replace(' ', '_') in original.lower()
        }

        if not filtered_crops:
            st.sidebar.warning(f"No models found for the '{selected_season}' season.")
        else:
            selected_clean_name = st.sidebar.selectbox(
                f"Select {selected_season} Crop to Predict", 
                list(filtered_crops.keys())
            )
            
            selected_crop_original = filtered_crops[selected_clean_name]
            sanitized_selected_crop = selected_crop_original.replace('/', '_')

            try:
                with open(f'saved_features/{sanitized_selected_crop}_features.json', 'r') as f:
                    feature_info = json.load(f)
                model_specific_features = feature_info['features']
                state_columns = feature_info['state_columns']
                user_input_features = [f for f in model_specific_features if not f.startswith('State_')]
            except FileNotFoundError:
                st.error(f"Feature information for '{selected_clean_name}' not found.")
                st.stop()
                
            all_states = sorted([s.replace('State_', '') for s in state_columns])
            all_states = [x.title() for x in all_states]
            selected_state = st.sidebar.selectbox("Select State", all_states, key=f"state_select_{sanitized_selected_crop}")

            st.sidebar.markdown("---")
            st.sidebar.markdown("**Enter Relevant Feature Values:**")
            
            input_data = {}
            for feature in user_input_features:
                input_data[feature] = st.sidebar.number_input(
                    clean_name(feature), 
                    value=0.0, 
                    format="%.2f", 
                    step=100.0,
                    min_value=0.0,
                    key=f"{sanitized_selected_crop}_{feature}"
                )


            if st.sidebar.button("Predict Yield", type="primary"):
                all_input_values = list(input_data.values())
                
                if sum(all_input_values) <= 0:
                    st.warning("Please enter at least one non-zero feature value to make a prediction.")
                else:
                    final_features_dict = {feature: 0.0 for feature in model_specific_features}
                    final_features_dict.update(input_data)
                    
                    selected_state_col_name = f'State_{selected_state}'
                    if selected_state_col_name in final_features_dict:
                        final_features_dict[selected_state_col_name] = 1
                    
                    prediction_df_final = pd.DataFrame([final_features_dict])[model_specific_features]

                    try:
                        model = joblib.load(f'saved_models/{sanitized_selected_crop}_Ensemble (Voting).joblib')
                        prediction = model.predict(prediction_df_final)
                        
                        st.subheader(f"Ensemble Model Prediction for {selected_clean_name}")
                        if prediction[0] < 0:
                            st.warning("Prediction is not possible. The model may not be reliable for the provided input values.")
                        else:
                            # --- FIX: Calculate delta for high/low indicator ---
                            avg_yield = average_yields.get(selected_crop_original, 0)
                            delta = prediction[0] - avg_yield
                            
                            st.metric(
                                label="Predicted Yield", 
                                value=f"{prediction[0]:.2f} kg/Ha",
                                delta=f"vs. Average"
                            )
                            st.info("This prediction is an average from multiple models for improved accuracy and stability.")
                    except FileNotFoundError:
                         st.error(f"The 'Ensemble (Voting)' model for the selected crop could not be found.")

    # --- TAB 2: EXPLORATORY DATA ANALYSIS (EDA) ---
    with tab2:
        st.header("Exploratory Data Analysis")
        st.markdown("Create quick charts to explore the relationships in the dataset.")

        # Separate columns into numeric and categorical for easier selection
        numeric_cols = eda_df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = ['State', 'District']

        chart_type = st.selectbox("Select Chart Type", ["Scatter Plot", "Bar Chart"])

        if chart_type == "Scatter Plot":
            st.markdown("Explore the relationship between two variables.")
            x_col = st.selectbox("Select X-axis variable", numeric_cols, index=numeric_cols.index('Year'))
            y_col = st.selectbox("Select Y-axis variable", numeric_cols)
            fig = px.scatter(eda_df, x=x_col, y=y_col, title=f"{clean_name(y_col)} vs. {clean_name(x_col)}")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar Chart":
            st.markdown("Compare a value across different categories.")
            cat_col = st.selectbox("Select a category (X-axis)", categorical_cols)
            num_col = st.selectbox("Select a value to compare (Y-axis)", numeric_cols)
            
            # Group by the category and calculate the mean for the bar chart
            bar_data = eda_df.groupby(cat_col)[num_col].mean().reset_index().sort_values(by=num_col, ascending=False)
            
            fig = px.bar(bar_data, x=cat_col, y=num_col, title=f"Average {clean_name(num_col)} by {clean_name(cat_col)}")
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: EXPLANATION ---
    with tab3:
        st.header("About This Application")
        st.markdown("""
        This application is designed to be a comprehensive tool for analyzing and predicting crop yields across various states and districts in India. It is built upon a dataset that combines information on crop area, irrigation, and yield.

        ### How the Predictor Works
        The prediction engine uses a sophisticated **Ensemble (Voting) Model**. Here's what that means:
        
        1.  **Multiple Models**: We first train several different types of machine learning models, including Linear Regression, Decision Trees, and Random Forests.
        2.  **Collective Wisdom**: The Ensemble model combines the predictions from the Linear Regression and Random Forest models.
        3.  **Averaging for Accuracy**: It takes the average of their predictions to produce a single, final output. This technique often leads to more stable and reliable results than relying on a single model alone, as it balances out the individual strengths and weaknesses of each component model.
        4.  **Context-Aware Features**: For each crop, the models are trained using only the most relevant features. This includes data specific to that crop (like its planted area) and general factors like the total sown area in the region.

        ### How to Use the EDA Tab
        The **Exploratory Data Analysis (EDA)** tab allows you to investigate the raw data yourself. You can:
        -   **Create Histograms** to see the frequency and distribution of different data points (e.g., how common certain yield values are).
        -   **Generate Scatter Plots** to see if there's a relationship between two variables (e.g., does yield increase as area increases?).
        -   **Build Bar Charts** to compare average values across different categories (e.g., which state has the highest average yield for a particular crop?).
        
        This tool is designed to provide both predictive power and data-driven insights.
        """)
