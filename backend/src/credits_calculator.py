import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

class CreditsCalculator:
    def __init__(self, historical_df, roles_df):
        """
        Initializes the calculator with historical performance and player roles
        
        Args:
            historical_df (pd.DataFrame): DataFrame with columns ['player_id', 'match_id', 'actual_fp']
            roles_df (pd.DataFrame): DataFrame with columns ['player_id', 'role']
        """
        self.historical_df = historical_df
        self.roles_df = roles_df.drop_duplicates(subset='player_id', keep='last').set_index('player_id')

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

    def calculate_all_player_credits(self):
        """
        Calculates the pre-match composite score for ALL players based on their full history
        This is needed to create the percentile pools.
        """
        player_credits_data = []
        
        for player_id in self.historical_df['player_id'].unique():
            history = self.historical_df[self.historical_df['player_id'] == player_id]['actual_fp']
            appearances = len(history)
            
            # Use last 10 appearances for composite score calculation
            last_10_fp = history.tail(10)
            composite_score = self._calculate_composite_score(last_10_fp)
            
            try:
                role = self.roles_df.loc[player_id, 'role']
            except KeyError:
                role = 'BAT' # Default role
            
            player_credits_data.append({
                'player_id': player_id,
                'role': role,
                'appearances': appearances,
                'composite_score': composite_score
            })
            
        return pd.DataFrame(player_credits_data)

    def get_credits_for_squad(self):
        """
        The main function to orchestrate the credit calculation for an entire player pool
        """
        all_players_df = self.calculate_all_player_credits()
        
        # --- Separate experienced players from newcomers ---
        experienced_players = all_players_df[all_players_df['appearances'] >= 10].copy()
        newcomers = all_players_df[all_players_df['appearances'] < 10].copy()

        # --- Calculate credits for experienced players ---
        credit_bands = {
            'top': {'range': (10.5, 11.0), 'percentile': (90, 100)},
            'next': {'range': (9.0, 10.0), 'percentile': (70, 90)},
            'middle': {'range': (7.0, 8.5), 'percentile': (30, 70)},
            'bottom': {'range': (4.0, 6.5), 'percentile': (0, 30)}
        }
        
        final_credits = []
        role_medians = {}

        for role in experienced_players['role'].unique():
            role_df = experienced_players[experienced_players['role'] == role].copy()
            scores = role_df['composite_score']
            
            if scores.empty: continue

            # Calculate percentiles for all players in this role
            role_df.loc[:, 'percentile'] = role_df['composite_score'].apply(lambda x: percentileofscore(scores, x))
            
            def get_credit(p):
                for band in credit_bands.values():
                    min_p, max_p = band['percentile']
                    if min_p <= p < max_p:
                        min_c, max_c = band['range']
                        # Linear interpolation
                        pos_in_band = (p - min_p) / (max_p - min_p)
                        return min_c + pos_in_band * (max_c - min_c)
                return 11.0 # For the 100th percentile case

            role_df.loc[:, 'credits'] = role_df['percentile'].apply(get_credit)
            final_credits.append(role_df[['player_id', 'credits']])
            
            role_medians[role] = role_df['credits'].median()

        experienced_credits = pd.concat(final_credits)

        # --- Calculate credits for newcomers ---
        def get_newcomer_credit(role):
            median = role_medians.get(role, 7.5) # Default median if role has no experienced players
            return np.clip(median, median - 0.5, median + 0.5)

        newcomers['credits'] = newcomers['role'].apply(get_newcomer_credit)
        
        # --- Combine and finalize ---
        all_credits = pd.concat([experienced_credits, newcomers[['player_id', 'credits']]])
        all_credits['credits'] = all_credits['credits'].round(2)
        all_credits['credits'] = np.clip(all_credits['credits'], 4.0, 11.0)
        
        return all_credits.set_index('player_id')

# --- Test the module ---
if __name__ == '__main__':
    DATASET_PATH = 'Dream11_Prod_Dev/backend/data/training_dataset.csv'
    
    # Generated dataset as the source of historical performance
    full_dataset = pd.read_csv(DATASET_PATH)
    
    # The historical_df needs player_id, match_id, and actual_fp
    historical_performance = full_dataset[['player_id', 'match_id', 'actual_fp']].sort_values(by='match_id')
    
    # The roles_df needs player_id and role
    player_roles = full_dataset[['player_id', 'role']]

    # Initialize and run the calculator
    calculator = CreditsCalculator(historical_performance, player_roles)
    credits_df = calculator.get_credits_for_squad()

    print("--- Sample of Calculated Player Credits ---")

    # Join with names for readability
    player_names = full_dataset[['player_id', 'player_name']].drop_duplicates().set_index('player_id')
    final_df = credits_df.join(player_names).sort_values(by='credits', ascending=False)
    
    print(final_df.head(10))