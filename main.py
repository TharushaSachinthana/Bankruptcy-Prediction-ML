from src.preprocess import preprocess_data
from src.feature_selection import select_features
from src.train_models import train_and_evaluate

RAW_DATA_PATH = "data/raw_data.csv"
PROCESSED_DATA_PATH = "data/processed_data.csv"
FEATURES_PATH = "results/selected_features.txt"
RESULTS_PATH = "results/model_comparison.csv"

# Step 1: Preprocess data
X, y = preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)

# Step 2: Feature selection
X_selected = select_features(X, y, FEATURES_PATH)

# Step 3: Train and evaluate models
train_and_evaluate(X_selected, y, RESULTS_PATH)
