import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from datetime import datetime

from src.data_processor import PlayerDataProcessor

class CreditsCalculator:
    """
    Calculates player credits for a given match in a fully time-aware manner.
    Preventing any data leakage from future matches.
    """
    def __init__(self, historical_df, roles_processor):
        """
        Initializes the calculator with historical performance and player roles
        
        Args:
            historical_df (pd.DataFrame): The complete historical dataset.
            roles_processor (PlayerDataProcessor): The roles processor instance.
        """
        self.historical_df = historical_df
        self.historical_df['match_date'] = pd.to_datetime(self.historical_df['match_date'])
        self.roles_processor = roles_processor

    def _calculate_composite_score(self, fp_series):
        """Calculates the composite score from a series of the last 10 fantasy points."""
        if len(fp_series) < 1:
            return 0
        
        mu_fp_10 = fp_series.mean()
        std_fp_10 = fp_series.std()
        
        # Per rule: if n=1, set σ=0. pandas .std() of a single value is NaN
        if pd.isna(std_fp_10):
            std_fp_10 = 0
            
        composite_score = 0.7 * mu_fp_10 + 0.3 * (mu_fp_10 - std_fp_10)
        return composite_score
    
    def get_credits_for_match(self, squad_player_ids, match_date):
        """
        Calculates credits for a specific squad for a match on a given date.
        """
        match_date_dt = pd.to_datetime(match_date)

        # 1. Filter history to only include matches STRICTLY BEFORE the match_date
        pre_match_history = self.historical_df[self.historical_df['match_date'] < match_date_dt].copy()

        # Handle the edge case of the very first match in the dataset
        if pre_match_history.empty:
            return pd.DataFrame({'credits': [6.0] * len(squad_player_ids)}, index=squad_player_ids)
        
        # 2. Calculate appearances and composite scores for ALL players in history
        player_stats = []
        for player_id, group in pre_match_history.groupby('player_id'):
            appearances = len(group)
            last_10_fp = group['actual_fp'].tail(10)
            composite_score = self._calculate_composite_score(last_10_fp)
            player_stats.append({
                'player_id': player_id,
                'appearances': appearances,
                'composite_score': composite_score
            })

        stats_df = pd.DataFrame(player_stats)
        # Assign roles based on the current match_date
        stats_df['role'] = stats_df['player_id'].apply(lambda pid: self.roles_processor.get_player_role(pid, match_date))

        # 3. Create the correct percentile pool (>= 10 prior appearances)
        percentile_pool = stats_df[stats_df['appearances'] >= 10].copy()
        
        credit_bands = {
            'top': {'range': (10.5, 11.0), 'percentile': (90, 100)},
            'next': {'range': (9.0, 10.0), 'percentile': (70, 90)},
            'middle': {'range': (7.0, 8.5), 'percentile': (30, 70)},
            'bottom': {'range': (4.0, 6.5), 'percentile': (0, 30)}
        }
        
        role_median_credits = {}
        processed_credits = []

        for role in percentile_pool['role'].unique():
            role_pool = percentile_pool[percentile_pool['role'] == role]
            scores = role_pool['composite_score']
            
            if scores.empty: continue
            
            def get_credit(p):
                for band in credit_bands.values():
                    min_p, max_p = band['percentile']
                    if min_p <= p < max_p:
                        min_c, max_c = band['range']
                        # Linear interpolation
                        pos_in_band = (p - min_p) / (max_p - min_p) if (max_p - min_p) > 0 else 0
                        return min_c + pos_in_band * (max_c - min_c)
                return 11.0 # For the 100th percentile case
            
            role_pool = role_pool.copy()
            role_pool['percentile'] = role_pool['composite_score'].apply(lambda x: percentileofscore(scores, x))
            role_pool['credits'] = role_pool['percentile'].apply(get_credit)
            role_median_credits[role] = role_pool['credits'].median()
            processed_credits.append(role_pool[['player_id', 'credits']])
        
        # Consolidate experienced player credits
        final_credits_df = pd.concat(processed_credits) if processed_credits else pd.DataFrame(columns=['player_id', 'credits'])
        
        # 4. Assign credits to the players in the current match's squad
        squad_credits = []
        for player_id in squad_player_ids:
            player_appearances = stats_df.loc[stats_df['player_id'] == player_id, 'appearances'].iloc[0] if player_id in stats_df['player_id'].values else 0
            player_role = self.roles_processor.get_player_role(player_id, match_date)

            credit_val = 0
            if player_appearances >= 10:
                # Find credit from the processed list of experienced players
                credit_entry = final_credits_df[final_credits_df['player_id'] == player_id]
                if not credit_entry.empty:
                    credit_val = credit_entry['credits'].iloc[0]
            else:
                # Assign newcomer credit based on the role's median
                credit_val = role_median_credits.get(player_role, 7.5) # Default to 7.5 if role has no median
            
            squad_credits.append({'player_id': player_id, 'credits': credit_val})

        result_df = pd.DataFrame(squad_credits).set_index('player_id')
        result_df['credits'] = result_df['credits'].round(2).clip(4.0, 11.0)
        return result_df

# --- Test the module ---
if __name__ == '__main__':

    SEASONAL_ROLES_PATH = 'Dream11_Prod_Dev/backend/data/player_roles_by_season.csv'
    GLOBAL_ROLES_PATH = 'Dream11_Prod_Dev/backend/data/player_roles_global.csv'
    HISTORICAL_DF_PATH = 'Dream11_Prod_Dev/backend/data/training_dataset.csv'

    
    # 1. Load the necessary assets
    print("Loading historical data and roles processor...")
    try:
        historical_df = pd.read_csv(HISTORICAL_DF_PATH)
        roles_processor = PlayerDataProcessor(SEASONAL_ROLES_PATH, GLOBAL_ROLES_PATH)
        print("Assets loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure you have run the feature pipeline to generate 'training_dataset.csv'.")
        exit()

    # 2. Initialize the calculator
    calculator = CreditsCalculator(historical_df, roles_processor)
    print("CreditsCalculator initialized.")

    # 3. Select a sample match from the validation set to test
    # We pick a match from later in the dataset to ensure there's history
    sample_match = historical_df[historical_df['match_id'] == historical_df['match_id'].unique()[-50]]
    test_match_date = sample_match['match_date'].iloc[0]
    test_squad_ids = sample_match['player_id'].tolist()
    
    print(f"\nTesting credit calculation for a sample match on: {test_match_date}")
    print(f"Squad size: {len(test_squad_ids)} players")

    # 4. Run the time-aware credit calculation
    credits_for_match_df = calculator.get_credits_for_match(test_squad_ids, test_match_date)

    # 5. Display the results
    print("\n--- Calculated Credits for Sample Match Squad ---")
    # Join with names for readability
    player_names = sample_match[['player_id', 'player_name']].drop_duplicates().set_index('player_id')
    final_df = credits_for_match_df.join(player_names).sort_values(by='credits', ascending=False)
    
    print(final_df.head(15))