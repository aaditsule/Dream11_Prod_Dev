# Project Challenges & Unique Selling Points (USPs)

## Challenges Faced

Building this application involved overcoming several technical challenges:

1.  **Preventing Data Leakage**: The most critical challenge was ensuring that the model did not learn from future data. This was solved by implementing a strict, chronological data processing pipeline where features for any given match were calculated using only data from prior matches.

2.  **Robust Data Handling**: The backend pipeline involved merging data from multiple sources and performing complex calculations. We encountered and solved several Pandas-related bugs, including `NaN` value propagation, mismatched data types (`KeyError`), and overlapping column names (`ValueError`). The final application is hardened against these issues.

3.  **Constrained Optimization**: Implementing the team selection rules required using a specialized linear programming library (`PuLP`). We successfully translated all game rules (budget, roles, team caps) into mathematical constraints to guarantee every recommended team is valid.

## Unique Selling Points (USPs)

Our Dream Team Predictor stands out for the following reasons:

1.  **Explainable AI (XAI)**: A key feature of our application is the "Show Rationale" button. It uses SHAP (SHapley Additive exPlanations) to provide a clear, visual breakdown of which features contributed to each player's predicted score. This builds user trust by making the model's decisions transparent, moving it from a "black box" to an understandable tool.

2.  **Guaranteed Valid Teams**: By using a `PuLP`-based constraints solver, our application guarantees that every single team it recommends is 100% compliant with all of Dream11's complex rules. It also handles "infeasible" scenarios gracefully by providing a clear error message.

3.  **End-to-End Automation**: The application provides a seamless user experience, going from a raw match data file to a fully optimized and explained fantasy team with a single click.

4.  **Modern & Performant Tech Stack**: The use of a FastAPI backend and a React frontend ensures a fast, responsive, and modern user experience, capable of handling complex calculations efficiently.