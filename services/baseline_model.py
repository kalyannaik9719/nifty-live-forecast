import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = Path("data/processed/nifty_features.parquet")
MODEL_PATH = Path("data/processed/baseline_model.pkl")

FEATURES = [
    "ret1",
    "ret5",
    "sma5",
    "sma20",
    "ema12",
    "volatility",
    "close_sma5_gap",
    "close_sma20_gap",
    "range",
    "hour",
    "minute"
]

def load_data():
    df = pd.read_parquet(DATA_PATH)
    return df

def train_model(df):

    X = df[FEATURES]
    y = df["target_up"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("Model Accuracy:", acc)

    return model

def save_model(model):

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model saved →", MODEL_PATH)

if __name__ == "__main__":

    df = load_data()
    model = train_model(df)
    save_model(model)