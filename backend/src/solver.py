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
        self.players_df = players_df.set_index('player_id')
        self.player_ids = self.players_df.index.tolist()
        self.teams = self.players_df['team'].unique().tolist()
        self.roles = self.players_df['role'].unique().tolist()

    def select_team(self):
        # 1. Define the problem: We want to maximize the total predicted fantasy points
        prob = LpProblem("FantasyTeamSelection", LpMaximize)

        # 2. Define decision variables: A binary variable for each player
        player_vars = LpVariable.dicts("Player", self.player_ids, cat=LpBinary)

        # 3. Define the objective function
        prob += lpSum(
            self.players_df.loc[p_id, 'predicted_fp'] * player_vars[p_id] - 
            0.001 * self.players_df.loc[p_id, 'credits'] * player_vars[p_id]
            for p_id in self.player_ids
        ), "TotalPredictedPoints"

        # 4. Add Constraints
        # Total players must be 11
        prob += lpSum(player_vars[p_id] for p_id in self.player_ids) == 11, "TotalPlayers"

        # Budget constraint (<= 100 credits)
        prob += lpSum(self.players_df.loc[p_id, 'credits'] * player_vars[p_id] for p_id in self.player_ids) <= 100, "TotalCredits"
        
        # Role constraints
        role_map = {role: [p_id for p_id in self.player_ids if self.players_df.loc[p_id, 'role'] == role] for role in self.roles}
        prob += lpSum(player_vars[p_id] for p_id in role_map.get('WK', [])) >= 1, "MinWicketKeepers"
        prob += lpSum(player_vars[p_id] for p_id in role_map.get('WK', [])) <= 4, "MaxWicketKeepers"
        prob += lpSum(player_vars[p_id] for p_id in role_map.get('BAT', [])) >= 3, "MinBatsmen"
        prob += lpSum(player_vars[p_id] for p_id in role_map.get('BAT', [])) <= 6, "MaxBatsmen"
        prob += lpSum(player_vars[p_id] for p_id in role_map.get('AR', [])) >= 1, "MinAllRounders"
        prob += lpSum(player_vars[p_id] for p_id in role_map.get('AR', [])) <= 4, "MaxAllRounders"
        prob += lpSum(player_vars[p_id] for p_id in role_map.get('BOWL', [])) >= 3, "MinBowlers"
        prob += lpSum(player_vars[p_id] for p_id in role_map.get('BOWL', [])) <= 6, "MaxBowlers"

        # Team constraints (max 7 from one team, min 1 from each)
        for team in self.teams:
            team_players = [p_id for p_id in self.player_ids if self.players_df.loc[p_id, 'team'] == team]
            prob += lpSum(player_vars[p_id] for p_id in team_players) <= 7, f"MaxPlayersFrom_{team}"
            prob += lpSum(player_vars[p_id] for p_id in team_players) >= 1, f"MinPlayersFrom_{team}"

        # 5. Solve the problem
        prob.solve()

        # 6. Extract the results
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