import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
import joblib
import json
import os

print("--- Starting Model Training and Saving Process ---")

# Create directories to save the models and features if they don't exist
os.makedirs('saved_models', exist_ok=True)
os.makedirs('saved_features', exist_ok=True)

# 1. Load your dataset
try:
    df = pd.read_csv("Cleaned_merged_suffixed.csv")
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'Cleaned_merged_suffixed.csv' not found. Exiting.")
    exit()

# 2. Setup the modeling environment
all_yield_columns = [col for col in df.columns if 'Yield' in col]
print(f"Found {len(all_yield_columns)} yield columns to model.")

# Save the list of all crops for the app's dropdown menu
with open('saved_features/all_crop_targets.json', 'w') as f:
    json.dump(all_yield_columns, f)

models_to_train = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'Ensemble (Voting)': VotingRegressor(estimators=[('LR', LinearRegression()), ('RF', RandomForestRegressor(random_state=42, n_jobs=-1))]),
    'Neural Network': MLPRegressor(random_state=42, max_iter=500, hidden_layer_sizes=(64, 32))
}

# 3. Main Loop: Iterate, Train, and Save for each crop
for target_column in all_yield_columns:
    print(f"\n--- Processing Target: {target_column} ---")
    
    # --- FIX: Sanitize the filename to remove invalid characters ---
    sanitized_target_name = target_column.replace('/', '_')

    # Filter data for non-zero yields
    df_filtered = df[df[target_column] > 0].copy()
    if len(df_filtered) < 50:
        print(f"Skipping: Only {len(df_filtered)} non-zero samples found.")
        continue

    y = df_filtered[target_column]

    # One-Hot Encoding for 'State'
    if 'State' in df_filtered.columns:
        df_encoded = pd.get_dummies(df_filtered, columns=['State'], prefix='State')
    else:
        df_encoded = df_filtered

    # Rule-Based Feature Selection
    base_crop_name = '_'.join(target_column.split('_')[:-2])
    state_columns = [col for col in df_encoded.columns if col.startswith('State_')]
    general_feature_keywords = ['Net Area Sown', 'Total Cropped Area']
    
    identifiers_to_drop = [col for col in ['District', 'Year'] if col in df_encoded.columns]
    features_to_drop = all_yield_columns + identifiers_to_drop
    all_potential_features = df_encoded.drop(columns=features_to_drop, errors='ignore').columns

    relevant_features = []
    for feature in all_potential_features:
        if base_crop_name in feature or any(keyword in feature for keyword in general_feature_keywords):
            relevant_features.append(feature)

    relevant_features.extend(state_columns)
    relevant_features = sorted(list(set(relevant_features)))

    if not relevant_features:
        print(f"Skipping: No relevant features found.")
        continue

    X_selected = df_encoded[relevant_features].copy()
    print(f"Selected and training on {len(X_selected.columns)} features.")

    # Save the feature list and state columns for this crop using the sanitized name
    feature_info = {
        'features': relevant_features,
        'state_columns': state_columns
    }
    with open(f'saved_features/{sanitized_target_name}_features.json', 'w') as f:
        json.dump(feature_info, f)

    # Train and Save Each Model using the sanitized name
    for model_name, model in models_to_train.items():
        print(f"Training {model_name}...")
        model.fit(X_selected, y)
        
        # Save the trained model to a file
        joblib.dump(model, f'saved_models/{sanitized_target_name}_{model_name}.joblib')

print("\n--- ✅ All models have been trained and saved successfully! ---")
