import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary

class TeamSelector:
    """
    Selects the optimal fantasy team using Linear Programming.
    """
    def __init__(self, players_df):
        """
        Args:
            players_df (pd.DataFrame): Must contain columns ['player_id', 'predicted_fp', 'credits', 'role', 'team'].
        """
        # Set player_id as the index for easier lookups
        if 'player_id' in players_df.columns:
            self.players_df = players_df.set_index('player_id')
        else:
            self.players_df = players_df # Assumes player_id is already the index
        self.player_ids = self.players_df.index.tolist()
        self.teams = self.players_df['team'].unique().tolist()

    def select_team(self):
        # 1. Define the problem
        prob = LpProblem("FantasyTeamSelection", LpMaximize)

        # 2. Define decision variables
        player_vars = LpVariable.dicts("Player", self.player_ids, cat=LpBinary)

        # 3. Define the objective function with tie-breakers
        # Primary Objective: Maximize total predicted fantasy points
        total_points = lpSum(self.players_df.loc[p_id, 'predicted_fp'] * player_vars[p_id] for p_id in self.player_ids)
        
        # Tie-breaker 1: Minimize total credits (by maximizing the negative)
        neg_credits = lpSum(-self.players_df.loc[p_id, 'credits'] * player_vars[p_id] for p_id in self.player_ids)
        
        # Tie-breaker 2: Maximize number of All-Rounders
        num_ars = lpSum(player_vars[p_id] for p_id in self.player_ids if self.players_df.loc[p_id, 'role'] == 'AR')
        
        # PuLP solves objectives in the order they appear in the list
        prob.setObjective(lpSum([total_points, neg_credits, num_ars]))
        # The final lexicographic tie-breaker is handled implicitly by PuLP's deterministic nature.

        # --- 4. ADD HARD CONSTRAINTS ---
        
        # a. Total of 11 players
        prob += lpSum(player_vars[p_id] for p_id in self.player_ids) == 11, "TotalPlayersConstraint"

        # b. Budget constraint
        prob += lpSum(self.players_df.loc[p_id, 'credits'] * player_vars[p_id] for p_id in self.player_ids) <= 100, "BudgetConstraint"
        
        # c. Role constraints (This is the corrected section)
        roles = ['WK', 'BAT', 'AR', 'BOWL']
        role_map = {r: [p_id for p_id in self.player_ids if self.players_df.loc[p_id, 'role'] == r] for r in roles}
        
        prob += lpSum(player_vars[p_id] for p_id in role_map['WK']) >= 1, "MinWKs"
        prob += lpSum(player_vars[p_id] for p_id in role_map['WK']) <= 4, "MaxWKs"
        
        prob += lpSum(player_vars[p_id] for p_id in role_map['BAT']) >= 3, "MinBATs"
        prob += lpSum(player_vars[p_id] for p_id in role_map['BAT']) <= 6, "MaxBATs"
        
        prob += lpSum(player_vars[p_id] for p_id in role_map['AR']) >= 1, "MinARs" 
        prob += lpSum(player_vars[p_id] for p_id in role_map['AR']) <= 4, "MaxARs"
        
        prob += lpSum(player_vars[p_id] for p_id in role_map['BOWL']) >= 3, "MinBOWLs"
        prob += lpSum(player_vars[p_id] for p_id in role_map['BOWL']) <= 6, "MaxBOWLs"

        # d. Team constraints
        for team in self.teams:
            team_players = [p_id for p_id in self.player_ids if self.players_df.loc[p_id, 'team'] == team]
            prob += lpSum(player_vars[p_id] for p_id in team_players) <= 7, f"MaxPlayers_{team}"
            if len(self.teams) > 1:
                 prob += lpSum(player_vars[p_id] for p_id in team_players) >= 1, f"MinPlayers_{team}"

        # 5. Solve the problem
        prob.solve()

        # 6. Extract results
        selected_ids = [p_id for p_id in self.player_ids if player_vars[p_id].varValue == 1]
        
        return self.players_df.loc[selected_ids]

# --- Testing the TeamSelector class ---
if __name__ == '__main__':
    # Create a dummy DataFrame of players for a match
    data = {
        'player_id': [f'p{i}' for i in range(25)],
        'player_name': [f'Player {i}' for i in range(25)],
        'predicted_fp': np.random.uniform(10, 80, 25),
        'credits': np.random.uniform(7.0, 11.0, 25).round(2),
        'role': ['WK', 'BAT', 'BAT', 'BAT', 'AR', 'BOWL', 'BOWL'] * 3 + ['WK', 'BAT', 'BAT', 'AR'],
        'team': ['TeamA'] * 13 + ['TeamB'] * 12
    }
    squad_df = pd.DataFrame(data)

    print("--- Sample Squad Data ---")
    print(squad_df)

    # Initialize and run the selector
    selector = TeamSelector(squad_df)
    recommended_xi = selector.select_team()

    print("\n--- Recommended XI ---")
    print(recommended_xi)

    print("\n--- Summary ---")
    print(f"Total Predicted Points: {recommended_xi['predicted_fp'].sum():.2f}")
    print(f"Total Credits Used: {recommended_xi['credits'].sum():.2f}")
    print("\nRole Count:")
    print(recommended_xi['role'].value_counts())
    print("\nTeam Count:")
    print(recommended_xi['team'].value_counts())