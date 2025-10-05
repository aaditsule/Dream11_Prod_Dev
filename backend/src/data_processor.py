import pandas as pd
import json
from datetime import datetime

class PlayerDataProcessor:
    """
    Handles loading player role data and assigning roles to players for a given match
    """
    def __init__(self, seasonal_roles_path, global_roles_path):
        try:
            self.seasonal_roles_df = pd.read_csv(seasonal_roles_path)
            self.global_roles_df = pd.read_csv(global_roles_path).set_index('player_id')
            print("Successfully loaded player role CSV files.")
        except FileNotFoundError as e:
            print(f"Error loading CSV files: {e}")
            self.seasonal_roles_df = pd.DataFrame()
            self.global_roles_df = pd.DataFrame()

    def get_player_role(self, player_id, match_date):
        """
        Gets a player's role for a specific match
        It first tries to find a role for the specific season (year of the match)
        If not found, it falls back to the global role
        If still not found, it defaults to 'BAT'
        """
        season = datetime.strptime(match_date, '%Y-%m-%d').year
        
        # 1. Try to find seasonal role
        seasonal_role = self.seasonal_roles_df[
            (self.seasonal_roles_df['player_id'] == player_id) & 
            (self.seasonal_roles_df['season'] == season)
        ]
        
        if not seasonal_role.empty:
            return seasonal_role.iloc[0]['role']
            
        # 2. Fallback to global role
        try:
            global_role = self.global_roles_df.loc[player_id, 'role']
            return global_role
        except KeyError:
            # 3. Default for brand-new players
            return 'BAT' 

    def get_squads_with_roles(self, match_data):
        """
        Processes a match file and returns a dictionary of all players with their roles
        """
        squads_with_roles = {}
        match_date = match_data['info']['dates'][0]
        player_registry = match_data['info']['registry']['people']
        
        id_to_name_map = {v: k for k, v in player_registry.items()}

        all_players = list(player_registry.values())
        
        for player_id in all_players:
            role = self.get_player_role(player_id, match_date)
            squads_with_roles[player_id] = {
                'name': id_to_name_map.get(player_id, "Unknown"),
                'role': role
            }
            
        return squads_with_roles

# --- Testing the PlayerDataProcessor class ---
if __name__ == '__main__':
    # Paths to data files
    SEASONAL_ROLES_PATH = 'Dream11_Prod_Dev/backend/data/player_roles_by_season.csv'
    GLOBAL_ROLES_PATH = 'Dream11_Prod_Dev/backend/data/player_roles_global.csv'
    MATCH_FILE_PATH = 'Dream11_Prod_Dev/backend/data/sample_match.json'

    # Initialize the processor
    processor = PlayerDataProcessor(SEASONAL_ROLES_PATH, GLOBAL_ROLES_PATH)
    
    # Load the sample match file
    with open(MATCH_FILE_PATH, 'r') as f:
        data = json.load(f)
    
    # Get all players and their assigned roles
    players_with_roles = processor.get_squads_with_roles(data)
    
    print("\n--- Player Roles for Match 336002 ---")
    
    # Print the first 10 players for brevity
    for i, (player_id, player_info) in enumerate(players_with_roles.items()):
        if i < 10:
            print(f"{player_info['name']:<20} | Role: {player_info['role']}")