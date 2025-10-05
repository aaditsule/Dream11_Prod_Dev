import json
from collections import defaultdict

class FantasyPointsCalculator:
    """
    Calculates fantasy points for all players in a given match based on a fixed scoring table
    """
    
    SCORING_RULES = {
        'batting': {
            'run': 1,
            'boundary_bonus': 1,
            'six_bonus': 2,
            'duck': -2,
            'strike_rate': {
                (170, float('inf')): 6,
                (150, 169.99): 4,
                (130, 149.99): 2,
                (50, 69.99): -2,
                (0, 49.99): -4,
            }
        },
        'bowling': {
            'wicket': 25,
            'lbw_bowled_bonus': 8,
            '3_wickets': 6,
            '4_wickets': 10,
            '5_wickets': 16,
            'maiden_over': 12,
            'economy_rate': {
                (0, 5.0): 6,
                (5.01, 6.5): 4,
                (6.51, 8.0): 2,
                (10.0, 11.0): -2,
                (11.01, float('inf')): -4,
            }
        },
        'fielding': {
            'catch': 8,
            '3_catches_bonus': 4,
            'stumping': 12,
            'run_out_direct': 12,
            'run_out_shared': 6
        }
    }

    def __init__(self, match_data):
        self.match_data = match_data
        self.player_points = defaultdict(lambda: defaultdict(int))
        self.player_stats = defaultdict(lambda: defaultdict(int))
        self.player_registry = self._get_player_registry()

    def _get_player_registry(self):
        """Creates a mapping from player name to stable player_id"""
        registry = {}
        for team in self.match_data['info']['players']:
            for player_name in self.match_data['info']['players'][team]:
                player_id = self.match_data['info']['registry']['people'].get(player_name)
                if player_id:
                    registry[player_name] = player_id
        return registry

    def calculate_points(self):
        """Main function to process innings and calculate all points"""
        for innings in self.match_data['innings']:
            for over in innings['overs']:
                runs_in_over = 0
                for delivery in over['deliveries']:
                    self._process_delivery(delivery)
                    runs_in_over += delivery['runs']['total']
                
                # Maiden over bonus
                if runs_in_over == 0:
                    bowler_name = delivery['bowler']
                    bowler_id = self.player_registry.get(bowler_name)
                    if bowler_id:
                        self.player_points[bowler_id]['bowling'] += self.SCORING_RULES['bowling']['maiden_over']

        self._apply_end_of_match_bonuses()
        return self._format_final_points()

    def _process_delivery(self, delivery):
        """Processes a single ball, updating stats for batters, bowlers, and fielders"""
        batter_name = delivery['batter']
        bowler_name = delivery['bowler']
        batter_id = self.player_registry.get(batter_name)
        bowler_id = self.player_registry.get(bowler_name)

        # Batting points
        if batter_id:
            runs = delivery['runs']['batter']
            self.player_points[batter_id]['batting'] += runs * self.SCORING_RULES['batting']['run']
            if runs == 4:
                self.player_points[batter_id]['batting'] += self.SCORING_RULES['batting']['boundary_bonus']
            if runs == 6:
                self.player_points[batter_id]['batting'] += self.SCORING_RULES['batting']['six_bonus']
            
            # Update stats
            self.player_stats[batter_id]['runs_scored'] += runs
            if 'wides' not in delivery.get('extras', {}):
                 self.player_stats[batter_id]['balls_faced'] += 1

        # Bowling & Fielding points
        if 'wickets' in delivery:
            for wicket in delivery['wickets']:
                # Wicket points
                if bowler_id and wicket['kind'] != 'run out':
                    self.player_points[bowler_id]['bowling'] += self.SCORING_RULES['bowling']['wicket']
                    self.player_stats[bowler_id]['wickets'] += 1
                    if wicket['kind'] in ['bowled', 'lbw']:
                        self.player_points[bowler_id]['bowling'] += self.SCORING_RULES['bowling']['lbw_bowled_bonus']

                # Fielding points
                if 'fielders' in wicket:
                    for fielder in wicket['fielders']:
                        fielder_name = fielder['name']
                        fielder_id = self.player_registry.get(fielder_name)
                        if fielder_id:
                            if wicket['kind'] == 'run out':
                                if len(wicket['fielders']) == 1:
                                    self.player_points[fielder_id]['fielding'] += self.SCORING_RULES['fielding']['run_out_direct']
                                else:
                                    self.player_points[fielder_id]['fielding'] += self.SCORING_RULES['fielding']['run_out_shared']
                            elif wicket['kind'] == 'stumped':
                                 self.player_points[fielder_id]['fielding'] += self.SCORING_RULES['fielding']['stumping']
                            else: # Catch
                                self.player_points[fielder_id]['fielding'] += self.SCORING_RULES['fielding']['catch']
                                self.player_stats[fielder_id]['catches'] += 1
        
        # Bowling stats
        if bowler_id:
            total_runs_conceded = delivery['runs']['batter']
            if 'wides' in delivery.get('extras', {}): total_runs_conceded += delivery['extras']['wides']
            if 'noballs' in delivery.get('extras', {}): total_runs_conceded += delivery['extras']['noballs']
            self.player_stats[bowler_id]['runs_conceded'] += total_runs_conceded
            if 'wides' not in delivery.get('extras', {}) and 'noballs' not in delivery.get('extras', {}):
                self.player_stats[bowler_id]['legal_balls_bowled'] += 1

    def _apply_end_of_match_bonuses(self):
        """Applies bonuses that can only be calculated after the match is over."""
        for player_id, stats in self.player_stats.items():
            # Batting bonuses
            if stats['balls_faced'] >= 10:
                sr = (stats['runs_scored'] / stats['balls_faced']) * 100
                for (min_sr, max_sr), points in self.SCORING_RULES['batting']['strike_rate'].items():
                    if min_sr <= sr <= max_sr:
                        self.player_points[player_id]['batting'] += points
                        break
            
            # Duck
            if stats['runs_scored'] == 0 and stats['balls_faced'] > 0:
                self.player_points[player_id]['batting'] += self.SCORING_RULES['batting']['duck']

            # Bowling bonuses
            if stats['wickets'] == 3: self.player_points[player_id]['bowling'] += self.SCORING_RULES['bowling']['3_wickets']
            if stats['wickets'] == 4: self.player_points[player_id]['bowling'] += self.SCORING_RULES['bowling']['4_wickets']
            if stats['wickets'] == 5: self.player_points[player_id]['bowling'] += self.SCORING_RULES['bowling']['5_wickets']
            
            if stats['legal_balls_bowled'] >= 12:
                overs = stats['legal_balls_bowled'] / 6
                er = stats['runs_conceded'] / overs
                for (min_er, max_er), points in self.SCORING_RULES['bowling']['economy_rate'].items():
                    if min_er <= er <= max_er:
                        self.player_points[player_id]['bowling'] += points
                        break
            
            # Fielding bonuses
            if stats['catches'] >= 3:
                self.player_points[player_id]['fielding'] += self.SCORING_RULES['fielding']['3_catches_bonus']

    def _format_final_points(self):
        """Combines all point categories into a final total for each player."""
        final_scores = {}
        for player_id, points in self.player_points.items():
            total = sum(points.values())
            final_scores[player_id] = {
                'total_points': total,
                'breakdown': dict(points)
            }
        return final_scores

# --- Testing the FantasyPointsCalculator class ---
if __name__ == '__main__':
    # Load the sample match file
    file_path = 'Dream11_Prod_Dev/backend/data/sample_match.json' 
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Calculate points
    calculator = FantasyPointsCalculator(data)
    final_points = calculator.calculate_points()

    # Create a reverse mapping from player_id to player name for display
    id_to_name_map = {v: k for k, v in calculator.player_registry.items()}

    # Sort players by points and print the top 10
    sorted_players = sorted(final_points.items(), key=lambda item: item[1]['total_points'], reverse=True)
    
    print("--- Top 10 Fantasy Scorers for the Match ---")
    for player_id, points_data in sorted_players[:10]:
        player_name = id_to_name_map.get(player_id, "Unknown Player")
        print(f"{player_name:<20} | Total Points: {points_data['total_points']}")