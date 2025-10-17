# --- Imports ---
import pandas as pd
import joblib
import json
import numpy as np
import shap
import math
import os

# --- FastAPI Imports ---
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# --- Import Modules ---
from src.data_processor import PlayerDataProcessor
from src.credits_calculator import CreditsCalculator
from src.solver import TeamSelector

# --- Setup & Load Global Assets ---
logging.basicConfig(level=logging.INFO)
pulp_logger = logging.getLogger('pulp')
pulp_logger.setLevel(logging.ERROR)

print("Loading application assets...")

try:
    MODEL = joblib.load('model_artifacts/ProductUI_Model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found.")
    MODEL = None

try:
    HISTORICAL_DF = pd.read_csv('data/training_dataset.csv', parse_dates=['match_date'])
    print("Historical dataset loaded successfully.")
except FileNotFoundError:
    print("Error: training_dataset.csv not found.")
    HISTORICAL_DF = pd.DataFrame()

# --- Initialize Player Data Processor ---
SEASONAL_ROLES_PATH = 'data/player_roles_by_season.csv'
GLOBAL_ROLES_PATH = 'data/player_roles_global.csv'
PLAYER_PROCESSOR = PlayerDataProcessor(SEASONAL_ROLES_PATH, GLOBAL_ROLES_PATH)

# --- Initialize Credit Calculator ---
CREDIT_CALCULATOR = CreditsCalculator(HISTORICAL_DF, PLAYER_PROCESSOR)

# --- Create a SHAP explainer on startup ---
explainer = shap.TreeExplainer(MODEL)
print("SHAP explainer created.")

# --- Initialize FastAPI App ---
app = FastAPI(title="Dream11 Team Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def sanitize_for_json(obj):
    """
    Recursively traverses a data structure to convert non-JSON-compliant
    values (like NaN, numpy types) to JSON-compliant ones.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(elem) for elem in obj]
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    return obj

# --- API Endpoints ---
@app.post("/api/predict_team/")
async def create_prediction(file: UploadFile = File(...)):
    if not MODEL or HISTORICAL_DF.empty:
        raise HTTPException(status_code=500, detail="Server is not ready. Assets not loaded.")
    if file.content_type != "application/json":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a valid JSON file.")

   # --- 1. Data Preparation ---
    test_match_data = json.loads(await file.read())
    match_date = test_match_data['info']['dates'][0]
    teams = test_match_data['info']['teams']
    
    squads_with_roles = PLAYER_PROCESSOR.get_squads_with_roles(test_match_data)
    squad_df = pd.DataFrame.from_dict(squads_with_roles, orient='index').reset_index().rename(columns={'index': 'player_id'})

    # --- 2. Feature Generation ---
    pre_match_history = HISTORICAL_DF[HISTORICAL_DF['match_date'] < pd.to_datetime(match_date)]
    features_list = []
    for player_id in squad_df['player_id']:
        player_history = pre_match_history[pre_match_history['player_id'] == player_id]
        avg_fp_last_5 = player_history.tail(5)['actual_fp'].mean()
        matches_played = len(player_history)
        features_list.append({'player_id': player_id, 'avg_fp_last_5': avg_fp_last_5, 'matches_played': matches_played})
    
    features_df = pd.DataFrame(features_list)
    squad_df = pd.merge(squad_df, features_df, on='player_id')
    
    # --- 3. Prediction ---
    X_pred_pre_dummies = squad_df[['avg_fp_last_5', 'matches_played', 'role']]
    X_pred = pd.get_dummies(X_pred_pre_dummies, columns=['role'])

    for col in MODEL.feature_names_in_:
        if col not in X_pred.columns:
            X_pred[col] = 0
            
    squad_df['predicted_fp'] = MODEL.predict(X_pred[MODEL.feature_names_in_])

    # --- 4. Credit Calculation ---
    credits_df = CREDIT_CALCULATOR.get_credits_for_match(squad_df['player_id'].tolist(), match_date)
    squad_df = squad_df.join(credits_df, on='player_id')

    squad_df['team'] = squad_df['name'].apply(lambda name: teams[0] if name in test_match_data['info']['players'][teams[0]] else teams[1])
    squad_df = squad_df.fillna(0.0)

    # --- 5. Team Selection ---
    selector = TeamSelector(squad_df)
    recommended_xi = selector.select_team()
    
    if recommended_xi.empty:
        raise HTTPException(status_code=422, detail="Could not find a valid team. The squad might not meet role/team constraints or the budget is too tight.")

    # --- 6. Rationale Calculation ---
    # a. Set the index of the prediction features to align with the squad
    X_pred.index = squad_df['player_id']

    # b. Get SHAP values for the *entire* squad
    shap_values = explainer.shap_values(X_pred[MODEL.feature_names_in_])
    shap_df = pd.DataFrame(shap_values, columns=MODEL.feature_names_in_, index=X_pred.index)
    
    # c. Join the relevant SHAP values using a suffix to prevent column name conflicts
    xi_with_shap = recommended_xi.join(shap_df, rsuffix='_shap')
    print(xi_with_shap.columns)
    print("SHAP values joined with recommended XI:", xi_with_shap)
    
    # def get_rationale(row):
    #     return {feat: row[feat] for feat in MODEL.feature_names_in_}
    
    def get_rationale(row):
        # Use the suffixed columns to get the rationale values
        shap_cols = [f"{col}" for col in MODEL.feature_names_in_]
        # Create the dictionary with the original feature names as keys
        return {original_col: row[shap_col] for original_col, shap_col in zip(MODEL.feature_names_in_, shap_cols)}

    xi_with_shap['rationale'] = xi_with_shap.apply(get_rationale, axis=1)

    # --- 7. Final Formatting and Response ---
    final_xi_df = xi_with_shap.reset_index() # reset_index to get player_id as a column
    final_xi_df = final_xi_df[['name', 'role', 'team', 'credits', 'predicted_fp', 'rationale', 'player_id']]
    xi_records = final_xi_df.to_dict(orient='records')

    summary_data = {
        'total_predicted_points': recommended_xi['predicted_fp'].sum(),
        'total_credits_used': recommended_xi['credits'].sum(),
        'role_counts': recommended_xi['role'].value_counts().to_dict(),
        'team_counts': recommended_xi['team'].value_counts().to_dict()
    }
    
    response_data = {'recommended_xi': xi_records, 'summary': summary_data}
    
    # Apply the sanitizer to the final object before returning
    return sanitize_for_json(response_data)

# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Dream11 Team Predictor API"}