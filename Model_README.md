# Model Documentation

This document details the machine learning model used in the Dream Team Predictor application.

## 1. Model Choice

- **Final Model**: XGBoost Regressor (`xgboost.XGBRegressor`)
- **Rationale**: We chose XGBoost for its high performance on structured, tabular data. It is a gradient boosting algorithm that is known for its accuracy, speed, and ability to capture complex non-linear relationships between features. It consistently outperforms simpler baseline models like Linear Regression for this type of prediction task.

## 2. Feature Engineering

To predict a player's fantasy points, we engineered features based purely on their historical performance *before* the match date. The following features were used for the final model:

1.  **`avg_fp_last_5`**: The player's average fantasy points over their last 5 matches. This is a strong indicator of recent form.
2.  **`matches_played`**: The total number of matches the player has played. This helps the model distinguish between experienced players and newcomers.
3.  **`role`**: The player's assigned role (WK, BAT, AR, BOWL) for the match, which was one-hot encoded into separate binary features (`role_WK`, `role_BAT`, etc.). This is a fundamental feature that heavily influences a player's scoring potential.

## 3. Training Process

- **Validation Strategy**: To prevent data leakage and accurately simulate real-world prediction, a **time-aware validation split** was used. The full dataset of matches was sorted chronologically. The first 80% of the matches were used for training, and the final 20% were used for validation. This ensures the model is always tested on data that occurred after the data it was trained on.

- **Evaluation Metric**: The model's performance was evaluated using **Mean Absolute Error (MAE)**, which measures the average absolute difference between the predicted fantasy points and the actual fantasy points. Our final model achieved an MAE of approximately **21.55** on the validation set.

## 4. Inference Flow

When a new match JSON is uploaded, the inference pipeline performs the following steps:
1.  Generates the same set of features (`avg_fp_last_5`, `matches_played`, `role`) for every player in the match squads based on the historical dataset.
2.  Loads the pre-trained model artifact (`model_artifacts/ProductUI_Model.pkl`).
3.  Uses the model to predict the fantasy points for each player.
4.  These predictions are then passed to the constraints solver to select the final team.