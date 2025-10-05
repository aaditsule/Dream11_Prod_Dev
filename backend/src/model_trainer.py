import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

class ModelTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.features = ['avg_fp_last_5', 'matches_played', 'role_AR', 'role_BAT', 'role_BOWL', 'role_WK']
        self.target = 'actual_fp'

    def prepare_data(self):
        """Loads and prepares the data for training."""
        df = pd.read_csv(self.dataset_path)
        
        # --- Feature Engineering: One-Hot Encode the 'role' column ---
        # This converts the categorical 'role' into numerical columns the model can use
        df = pd.get_dummies(df, columns=['role'], prefix='role')
        
        # Ensure all possible role columns exist, even if one is missing in the split
        for col in self.features:
            if col not in df.columns:
                df[col] = 0

        # --- Time-Aware Split ---
        # We sort by match_id to simulate time and split the data.
        # The first 80% of matches will be for training, the last 20% for validation.
        df = df.sort_values(by='match_id')
        
        # Find the split point
        unique_matches = df['match_id'].unique()
        split_index = int(len(unique_matches) * 0.8)
        train_matches = unique_matches[:split_index]
        
        train_df = df[df['match_id'].isin(train_matches)]
        val_df = df[~df['match_id'].isin(train_matches)]

        X_train = train_df[self.features]
        y_train = train_df[self.target]
        X_val = val_df[self.features]
        y_val = val_df[self.target]

        return X_train, X_val, y_train, y_val

    def train(self):
        """Trains the XGBoost model and evaluates its performance."""
        X_train, X_val, y_train, y_val = self.prepare_data()

        print("Starting model training...")
        self.model.fit(X_train, y_train)
        print("Training complete")

        # Evaluate the model
        predictions = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, predictions)
        print(f"\nModel Performance on Validation Set (last 20% of matches):")
        print(f"Mean Absolute Error (MAE): {mae:.2f} fantasy points")
        
    def save_model(self, file_path):
        """Saves the trained model to a file."""
        print(f"\nSaving model to {file_path}")
        joblib.dump(self.model, file_path)
        print("Model saved successfully.")

# --- Run the training process ---
if __name__ == '__main__':
    DATASET_PATH = 'Dream11_Prod_Dev/backend/data/training_dataset.csv'
    MODEL_OUTPUT_PATH = 'Dream11_Prod_Dev/backend/model_artifacts/ProductUI_Model.pkl'

    trainer = ModelTrainer(DATASET_PATH)
    trainer.train()
    trainer.save_model(MODEL_OUTPUT_PATH)