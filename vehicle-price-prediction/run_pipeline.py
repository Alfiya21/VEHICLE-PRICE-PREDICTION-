from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from src.train_model import train_model
from src.evaluate_model import evaluate

DATA_PATH = "data/dataset.csv"

def main():
    df = load_and_clean_data(DATA_PATH)
    df = create_features(df)

    model, X_test, y_test = train_model(df)
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()
