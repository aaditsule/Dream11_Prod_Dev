import pandas as pd
import json
import os
from datetime import datetime
from tqdm import tqdm
import joblib

# Import project modules
from data_processor import PlayerDataProcessor
from fantasy_calculator import FantasyPointsCalculator
from credits_calculator import CreditsCalculator
from solver import TeamSelector

class EvaluationGenerator:
    def __init__(self, model_path, data_folder, roles_processor, historical_df):
        self.model = joblib.load(model_path)
        self.data_folder = data_folder
        self.roles_processor = roles_processor
        self.historical_df = historical_df
        self.validation_files = self._get_validation_files()

    def _get_validation_files(self):
        """Identifies the last 20% of matches to use as the validation set"""
        files_with_dates = []
        for filename in os.listdir(self.data_folder):
            if filename.endswith('.json'):
                file_path = os.path.join(self.data_folder, filename)
                with open(file_path, 'r') as f:
                    info = json.load(f)['info']
                    match_date = info['dates'][0]
                    files_with_dates.append((match_date, file_path))
        
        files_with_dates.sort(key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))
        
        split_index = int(len(files_with_dates) * 0.8)
        return [file[1] for file in files_with_dates[split_index:]]

    def process_match(self, match_file_path):
        """Processes a single match to get both the Dream XI and the Predicted XI"""
        with open(match_file_path, 'r') as f:
            match_data = json.load(f)

        match_id = os.path.splitext(os.path.basename(match_file_path))[0]
        match_date = match_data['info']['dates'][0]
        teams = match_data['info']['teams']

        # --- Base Player Info ---
        squads_with_roles = self.roles_processor.get_squads_with_roles(match_data)
        squad_df = pd.DataFrame.from_dict(squads_with_roles, orient='index').reset_index().rename(columns={'index': 'player_id'})
        squad_df['team'] = squad_df['name'].apply(lambda name: teams[0] if name in match_data['info']['players'][teams[0]] else teams[1])

        # --- 1. Calculate the DREAM XI (based on actual points) ---
        fp_calculator = FantasyPointsCalculator(match_data)
        actual_points = fp_calculator.calculate_points()
        actual_fp_df = pd.DataFrame([{'player_id': pid, 'actual_fp': data['total_points']} for pid, data in actual_points.items()])
        
        dream_squad_df = pd.merge(squad_df, actual_fp_df, on='player_id', how='left').fillna(0)
        
        # Need credits for the solver, even for the Dream XI
        credit_calculator = CreditsCalculator(self.historical_df[['player_id', 'match_id', 'actual_fp']], self.historical_df[['player_id', 'role']])
        credits_df = credit_calculator.get_credits_for_squad()
        dream_squad_df = dream_squad_df.join(credits_df, on='player_id').fillna(6.0)

        # Use 'actual_fp' as the objective for the solver
        dream_squad_df['predicted_fp'] = dream_squad_df['actual_fp'] 
        dream_xi_selector = TeamSelector(dream_squad_df)
        dream_xi = dream_xi_selector.select_team().sort_values(by='actual_fp', ascending=False)
        dream_xi_total = dream_xi['actual_fp'].sum()

        # --- 2. Calculate the PREDICTED XI (based on model predictions) ---
        features_list = []
        for player_id in squad_df['player_id']:
            player_history = self.historical_df[self.historical_df['player_id'] == player_id]
            avg_fp_last_5 = player_history.tail(5)['actual_fp'].mean() if not player_history.empty else 0
            matches_played = len(player_history)
            features_list.append({'player_id': player_id, 'avg_fp_last_5': avg_fp_last_5, 'matches_played': matches_played})
        
        features_df = pd.DataFrame(features_list)
        pred_squad_df = pd.merge(squad_df, features_df, on='player_id')
        
        X_pred = pd.get_dummies(pred_squad_df[['avg_fp_last_5', 'matches_played', 'role']], columns=['role'])
        for col in self.model.feature_names_in_:
            if col not in X_pred.columns: X_pred[col] = 0
        
        pred_squad_df['predicted_fp'] = self.model.predict(X_pred[self.model.feature_names_in_])
        pred_squad_df = pred_squad_df.join(credits_df, on='player_id').fillna(6.0)

        predicted_xi_selector = TeamSelector(pred_squad_df)
        predicted_xi = predicted_xi_selector.select_team().sort_values(by='predicted_fp', ascending=False)
        
        # Get the actual points of the players in the predicted team
        predicted_xi_with_actual_points = pd.merge(predicted_xi, actual_fp_df, on='player_id', how='left').fillna(0)
        predicted_xi_total = predicted_xi_with_actual_points['actual_fp'].sum()
        
        # --- 3. Calculate Metrics and Format Output ---
        ae_team_total = abs(dream_xi_total - predicted_xi_total)

        return {
            "match_id": match_id,
            "match_date": match_date,
            "team1": teams[0],
            "team2": teams[1],
            "predicted_xi": ", ".join(predicted_xi['name'].tolist()),
            "dream_xi": ", ".join(dream_xi['name'].tolist()),
            "predicted_points_per_player": ", ".join(map(str, predicted_xi['predicted_fp'].round(2).tolist())),
            "ae_team_total": ae_team_total
        }

    def generate_summary(self, output_path):
        """Runs the evaluation for all validation files and saves the summary."""
        results = []
        for match_file in tqdm(self.validation_files, desc="Generating Evaluation Summary"):
            try:
                result = self.process_match(match_file)
                results.append(result)
            except Exception as e:
                print(f"Could not process {os.path.basename(match_file)}: {e}")

        summary_df = pd.DataFrame(results)
        summary_df.to_csv(output_path, index=False)
        
        mean_ae = summary_df['ae_team_total'].mean()
        print(f"\nEvaluation summary saved to {output_path}")
        print(f"Leaderboard Metric (mean_ae_team_total): {mean_ae:.2f}")

# --- Testing the EvaluationGenerator class ---
if __name__ == '__main__':
    # Define paths
    MODEL_PATH = 'Dream11_Prod_Dev/backend/model_artifacts/ProductUI_Model.pkl'
    DATA_FOLDER = 'Dream11_Prod_Dev/backend/data/Input_Matches_DATA/'
    SEASONAL_ROLES_PATH = 'Dream11_Prod_Dev/backend/data/player_roles_by_season.csv'
    GLOBAL_ROLES_PATH = 'Dream11_Prod_Dev/backend/data/player_roles_global.csv'
    HISTORICAL_DF_PATH = 'Dream11_Prod_Dev/backend/data/training_dataset.csv'
    OUTPUT_PATH = 'Dream11_Prod_Dev/backend/output/eval_summary.csv'

    # Load assets
    roles_processor = PlayerDataProcessor(SEASONAL_ROLES_PATH, GLOBAL_ROLES_PATH)
    historical_df = pd.read_csv(HISTORICAL_DF_PATH)

    # Create the generator and run it
    eval_generator = EvaluationGenerator(MODEL_PATH, DATA_FOLDER, roles_processor, historical_df)
    eval_generator.generate_summary(OUTPUT_PATH)    