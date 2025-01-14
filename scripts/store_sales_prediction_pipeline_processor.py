import pandas as pd


# Data paths
train_data_path = "../data/train_cleaned_data.csv"
try:
    train_data = pd.read_csv(train_data_path)
    # test_data = pd.read_csv(test_data_path)
except FileNotFoundError as e:
    print(f"Error: {e}")

# import class from the module
from scripts.store_sales_prediction import StoreSalesPrediction


def store_sales_prediction_pipeline_processor():
    # Intializing the class
    train_data = pd.read_csv(train_data_path)
    predictor = StoreSalesPrediction(train_data)

    # preprocessing data
    predictor.preprocess_data()

    # Building RandomForest model
    pipeline = predictor.build_sklearn_model()
    # Save models using pickle
    predictor.save_randomforest_model()

    # Calculating and printing features
    predictor.feature_importance()

    # Building LSTM model using PyTorch
    lstm_model = predictor.build_LSTM_model()

    # Saving the LSTM model
    predictor.save_lstm_model()
