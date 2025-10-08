import pandas as pd
import json
import os
from datetime import datetime
from tqdm import tqdm

from fantasy_calculator import FantasyPointsCalculator
from data_processor import PlayerDataProcessor

class FeaturePipeline:
    def __init__(self, data_folder_path, roles_processor):
        self.data_folder_path = data_folder_path
        self.roles_processor = roles_processor
        self.match_files = self._get_sorted_match_files()
        self.historical_stats = pd.DataFrame(columns=['player_id', 'actual_fp'])

    def _get_sorted_match_files(self):
        """Gets all JSON file paths and sorts them chronologically"""
        files_with_dates = []
        for filename in os.listdir(self.data_folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(self.data_folder_path, filename)
                with open(file_path, 'r') as f:
                    info = json.load(f)['info']
                    match_date = info['dates'][0]
                    files_with_dates.append((match_date, file_path))
        
        # Sort by date
        files_with_dates.sort(key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))
        return [file[1] for file in files_with_dates]

    def create_dataset(self):
        """
        Processes all matches chronologically to build a feature-rich dataset
        """
        all_match_data = []

        for match_file in tqdm(self.match_files, desc="Processing Matches"):
            with open(match_file, 'r') as f:
                match_data = json.load(f)

            match_id = os.path.splitext(os.path.basename(match_file))[0]
            match_date = match_data['info']['dates'][0]
            
            fp_calculator = FantasyPointsCalculator(match_data)
            current_match_points = fp_calculator.calculate_points()

            players_with_roles = self.roles_processor.get_squads_with_roles(match_data)

            for player_id, role_info in players_with_roles.items():
                
                # --- FEATURE ENGINEERING ---
                avg_fp_last_5 = 0
                matches_played = 0
                
                if not self.historical_stats.empty:
                    player_history = self.historical_stats[self.historical_stats['player_id'] == player_id]
                    if not player_history.empty:
                        last_5_matches = player_history.tail(5)
                        avg_fp_last_5 = last_5_matches['actual_fp'].mean()
                        matches_played = len(player_history)
                
                actual_fp = current_match_points.get(player_id, {}).get('total_points', 0)

                all_match_data.append({
                    'match_id': match_id,
                    'match_date': match_date,
                    'player_id': player_id,
                    'player_name': role_info['name'],
                    'role': role_info['role'],
                    'avg_fp_last_5': avg_fp_last_5,
                    'matches_played': matches_played,
                    'actual_fp': actual_fp
                })

            # Update historical stats for the next iteration
            current_match_df = pd.DataFrame([
                {'player_id': pid, 'actual_fp': pdata.get('total_points', 0)}
                for pid, pdata in current_match_points.items()
            ])
            self.historical_stats = pd.concat([self.historical_stats, current_match_df], ignore_index=True)
            
        final_df = pd.DataFrame(all_match_data)
        final_df.drop_duplicates(subset=['match_id', 'player_id'], keep='last', inplace=True)

        return final_df

# --- Run the pipeline ---
if __name__ == '__main__':

    DATA_FOLDER = 'Dream11_Prod_Dev/backend/data/Input_Matches_DATA/'
    SEASONAL_ROLES_PATH = 'Dream11_Prod_Dev/backend/data/player_roles_by_season.csv'
    GLOBAL_ROLES_PATH = 'Dream11_Prod_Dev/backend/data/player_roles_global.csv'

    processor = PlayerDataProcessor(SEASONAL_ROLES_PATH, GLOBAL_ROLES_PATH)
    
    pipeline = FeaturePipeline(DATA_FOLDER, processor)
    training_dataset = pipeline.create_dataset()
    
    # Save the final dataset
    output_path = 'Dream11_Prod_Dev/backend/data/training_dataset.csv'

    training_dataset.to_csv(output_path, index=False)

    print(f"\nSuccessfully created dataset with {len(training_dataset)} rows")
    print(f"Saved to {output_path}")
    print("\nDataset Head:")
    print(training_dataset.head())