1) player_roles_by_season.csv — “per-season” roles

What it is:
A mapping for each (player_id, season) to a single role.

Why seasonal?
Some players change role profile across years (e.g., a batter starts bowling regularly and becomes an AR; a part-timer becomes a specialist bowler). Seasonal labels keep things accurate and time-aware.

Columns (typical):

player_id – stable ID from info.registry.people

season – year (e.g., 2019, 2020, …)

role – one of {WK, BAT, AR, BOWL}


How the label is decided (summary):

If a player has a stumped dismissal credited as fielder → WK.
Else, use batting + bowling usage thresholds to classify BOWL, AR, or BAT (deterministic rules).
Rules are applied per season using only matches in that season.

When to use it:
At inference time for a test match, compute season = year(match_date), then join on (player_id, season) to get the role. This is your first choice.

2) player_roles_global.csv — “global” fallback

What it is:
A single role per player_id, computed as a majority vote across seasons (with precedence for ties: WK > AR > BOWL > BAT).

Why have this?

A player may appear in a test match with no seasonal row (e.g., new season not present in training data).
You might have sparse history in some seasons.

When to use it:

If the seasonal lookup (player_id, season) fails, fall back to the global role.
If both are missing (completely new player_id):
	Default to BAT (safest), unless you explicitly detect WK in your dataset historically (stumped events).

Optionally, show a small UI warning “role defaulted”.