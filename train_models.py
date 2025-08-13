import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import warnings
import joblib
import json
import os

warnings.filterwarnings('ignore', category=UserWarning)


try:
    df = pd.read_csv("Cleaned_merged_suffixed.csv")
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'Cleaned_merged_suffixed.csv' not found.")
    df = pd.DataFrame()

if not df.empty:

    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('saved_features', exist_ok=True)


    all_yield_columns = [col for col in df.columns if 'Yield' in col]
    all_model_results = []
    successfully_trained_crops = []
    master_feature_set = set()


    models_to_test = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'Ensemble (Voting)': VotingRegressor(estimators=[('LR', LinearRegression()), ('RF', RandomForestRegressor(random_state=42, n_jobs=-1))]),
        'Neural Network': MLPRegressor(random_state=42, max_iter=500, hidden_layer_sizes=(64, 32))
    }
    

    for target_column in all_yield_columns:
        print(f"\n--- Processing Target: {target_column} ---")
        
        sanitized_target_name = target_column.replace('/', '_')

        df_filtered = df[df[target_column] > 0].copy()

        if len(df_filtered) < 1000:
            print(f"Skipping: Only {len(df_filtered)} non-zero samples found (min 1000).")
            continue
            
        y = df_filtered[target_column]

        # One-hot encoding for State
        if 'State' in df_filtered.columns:
            df_encoded = pd.get_dummies(df_filtered, columns=['State'], prefix='State')
            print("Applied one-hot encoding to the 'State' column.")
        else:
            df_encoded = df_filtered


        base_crop_name = '_'.join(target_column.split('_')[:-2])
        print(f"Base crop name identified as: '{base_crop_name}'")

        general_feature_keywords = ['Net Area Sown', 'Total Cropped Area', 'State_','Gross Irrigated Area Canal Total','Gross Irrigated Area Well Total','Gross Irrigated Area Other']
        
        identifiers_to_drop = [col for col in ['District', 'Year'] if col in df_encoded.columns]
        features_to_drop = all_yield_columns + identifiers_to_drop
        all_potential_features = df_encoded.drop(columns=features_to_drop, errors='ignore').columns

        relevant_features = []
        for feature in all_potential_features:
            if any(keyword in feature for keyword in general_feature_keywords):
                relevant_features.append(feature)
        
        if not relevant_features:
            print(f"Skipping: No relevant features found for '{target_column}'.")
            continue
        

        master_feature_set.update([f for f in relevant_features if not f.startswith('State_')])

        X_selected = df_encoded[list(set(relevant_features))].copy()
        print(f"Selected {len(X_selected.columns)} relevant features (including states).")
        

        feature_info = {
            'features': list(X_selected.columns),
            'state_columns': [col for col in X_selected.columns if col.startswith('State_')]
        }
        with open(f'saved_features/{sanitized_target_name}_features.json', 'w') as f:
            json.dump(feature_info, f)


        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        
        for model_name, model in models_to_test.items():
            model.fit(X_train, y_train)
            

            joblib.dump(model, f'saved_models/{sanitized_target_name}_{model_name}.joblib')
            
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)

            all_model_results.append({
                'Crop_Yield_Target': target_column,
                'Model': model_name,
                'R-squared': r2,
                'Number_of_Features': len(X_selected.columns)
            })
        
        successfully_trained_crops.append(target_column)


    with open('saved_features/trained_crop_models.json', 'w') as f:
        json.dump(successfully_trained_crops, f)
    with open('saved_features/master_feature_list.json', 'w') as f:
        json.dump(sorted(list(master_feature_set)), f)
        

    if all_model_results:
        print("\n--- Overall Model Performance (with State as a Feature) ---")
        results_df = pd.DataFrame(all_model_results)
        results_df_sorted = results_df.sort_values(by=['Crop_Yield_Target', 'R-squared'], ascending=[True, False])
        print(results_df_sorted)
    else:
        print("\nNo models were trained.")
