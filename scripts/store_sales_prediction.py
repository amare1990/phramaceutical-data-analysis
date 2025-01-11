""" Building ML and DL Models module for store sales prediction. """

import logging
import datetime
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# For PyTorch
from statsmodels.tsa.stattools import adfuller
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class StoreSalesPrediction:
    """A store sales prediction class"""

    def __init__(self, train_data: pd.DataFrame):
        self.data = train_data
        # self.test_data = test_data

        self.model = None
        self.scalar = StandardScaler()
        self.logger = self.setup_logger()

    @staticmethod
    def setup_logger():
        logger = logging.getLogger(f"StoreSalesPrediction")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        return logger

    def preprocess_data(self):
        """Preprocess train and test data."""
        self.logger.info(f"Starting preprocessing for building {self.model}...")

        # Strip white spaces in column names if there
        dataset = self.data
        dataset.columns = dataset.columns.str.strip()

        # Scale numerical columns (exclude 'Id' and 'Sales' if present)
        numeric_cols = dataset.select_dtypes(include=np.number).columns

        # Handle datetime columns
        dataset["Date"] = pd.to_datetime(dataset["Date"])
        dataset["Weekday"] = dataset["Date"].dt.weekday
        dataset["IsWeekend"] = dataset["Weekday"] >= 5
        dataset["Month"] = dataset["Date"].dt.month
        dataset["IsBeginningOfMonth"] = dataset["Date"].dt.day <= 10
        dataset["IsMidMonth"] = (dataset["Date"].dt.day > 10) & (
            dataset["Date"].dt.day <= 20
        )
        dataset["IsEndOfMonth"] = dataset["Date"].dt.day > 20
        dataset["Year"] = dataset["Date"].dt.year

        # One-hot encode the month segment features
        month_segment_encoded = pd.get_dummies(
            dataset[["IsBeginningOfMonth", "IsMidMonth", "IsEndOfMonth"]],
            drop_first=True,
        )
        dataset = pd.concat([dataset, month_segment_encoded], axis=1)

        current_year = dataset["Date"].dt.year
        current_week = dataset["Date"].dt.isocalendar().week

        # Calculate elapsed years and weeks for Promo2
        dataset["Promo2ElapsedYears"] = current_year - dataset["Promo2SinceYear"]
        dataset["Promo2ElapsedWeeks"] = current_week - dataset["Promo2SinceWeek"]
        dataset["StoreOpenMonths"] = (
            dataset["Date"].dt.year - dataset["CompetitionOpenSinceYear"]
        ) * 12 + (dataset["Date"].dt.month - dataset["CompetitionOpenSinceMonth"])
        dataset["IsPublicHoliday"] = (
            dataset["StateHoliday"].isin(["a", "b", "c"]).astype(int)
        )

        # One-hot encode StoreType and Assortment
        store_type_encoded = pd.get_dummies(dataset["StoreType"], prefix="StoreType")
        assortment_encoded = pd.get_dummies(dataset["Assortment"], prefix="Assortment")
        dataset = pd.concat([dataset, store_type_encoded, assortment_encoded], axis=1)
        dataset.drop(columns=["StoreType", "Assortment"], inplace=True)

        # Handle PromoInterval (One-Hot Encoding, including 'None' for
        # non-participating stores)
        if "PromoInterval" in dataset.columns:
            dataset["Promo2_Participation"] = (
                dataset["PromoInterval"] != "None"
            ).astype(int)
            promo_interval_encoded = pd.get_dummies(
                dataset["PromoInterval"], prefix="PromoInterval"
            )
            dataset = pd.concat([dataset, promo_interval_encoded], axis=1)
            dataset.drop(columns=["PromoInterval"], inplace=True)
        else:
            self.logger.warning(f"'PromoInterval' column is missing!")
            dataset["Promo2_Participation"] = 0  # Default value
            dataset.drop(columns=["PromoInterval"], inplace=True, errors="ignore")

        # Handle StateHoliday (One-Hot Encoding)
        stateholiday_encoded = pd.get_dummies(
            dataset["StateHoliday"], prefix="StateHoliday", drop_first=False
        )
        dataset = pd.concat([dataset, stateholiday_encoded], axis=1)
        dataset.drop(columns=["StateHoliday"], inplace=True, errors="ignore")

        # Assign self.data to dataset
        self.data = dataset

        self.logger.info(f"Preprocessing for building {self.model} model completed.")

        return self.data

    def build_sklearn_model(self):
        self.logger.info(f"Building sklearn model...")

        # Define the feature matrix (X) and target (y) using preprocessed data
        columns_to_drop = ["Date"]
        X = self.data.drop(
            columns=["Sales"] + columns_to_drop, axis=1, errors="ignore"
        )  # Drop 'Sales' and date-like features
        y = self.data["Sales"]

        # Split self.data into train and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define the feature matrix (X) and target (y) using preprocessed data
        # X_train = X_train.drop(['Sales', 'Date', 'Id'], axis=1)
        # y_train = y_train['Sales']

        # X_test = X_test.drop(['Sales', 'Date', 'Id'], axis=1)
        # y_test = y_test['Sales']

        # Check if the columns in train and test data are aligned
        assert all(
            X_train.columns == X_test.columns
        ), "Feature columns in train and test data do not match!"

        # Define the pipeline with a RandomForestRegressor
        pipeline = Pipeline(
            [("regressor", RandomForestRegressor(n_estimators=100, random_state=42))]
        )

        # Fit the model on the training data
        pipeline.fit(X_train, y_train)
        self.model = pipeline

        # Make predictions on the test set
        predictions = pipeline.predict(X_test)

        # Evaluate the model performance using RMSE and MAE
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)

        # Log the model performance metrics
        self.logger.info(f"Model RMSE: {rmse}")
        self.logger.info(f"Model MAE: {mae}")

        return pipeline

    # Feature importance implementation
    def feature_importance(self):
        if isinstance(self.model, Pipeline):
            model = self.model.named_steps["regressor"]
            importance = model.feature_importances_
            features = self.data.drop(["Sales", "Date"], axis=1).columns
            importance_df = pd.DataFrame(
                {"Feature": features, "Importance": importance}
            )
            importance_df.sort_values(by="Importance", ascending=False, inplace=True)

            self.logger.info(f"Feature Importance analysis for {self.model} model: ")
            self.logger.info(f"{importance_df}")

            sns.barplot(x="Importance", y="Feature", data=importance_df)
            plt.title("Feature Importance")
            plt.savefig("plots/feature_importance.png", dpi=300, bbox_inches="tight")
            plt.show()

    def save_model(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"Store_sales_model_{timestamp}.pkl"

        with open(filename, "wb") as file:
            pickle.dump(self.model, file)

        self.logger.info(f"Model saved as {filename}. ")

    class SalesDataset(Dataset):
        def __init__(self, data, look_back=1):
            self.data = data
            self.look_back = look_back
            self.X, self.y = self.create_supervised_data(data, look_back)

        def create_supervised_data(self, data, look_back=1):
            X, y = [], []
            for i in range(len(data) - look_back):
                X.append(data[i : i + look_back])
                y.append(data[i + look_back])
            return np.array(X), np.array(y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(
                self.y[idx], dtype=torch.float32
            )

    def build_LSTM_model(self):
        self.logger.info(f"Building LSTM {self.model} model using PyTorch...")

        # Convert into Time-series data
        sales_data = self.data[["Sales", "Date"]].set_index("Date")
        sales_data = sales_data.sort_index()

        # Check stationarity
        adf_test = adfuller(sales_data["Sales"])
        self.logger.info(f"ADF Test Statistic: {adf_test[0]}")
        self.logger.info(f"ADF Test p-value: {adf_test[1]}")

        # Prepare supervised learning data
        look_back = 7
        scaled_data = self.scalar.fit_transform(
            sales_data["Sales"].values.reshape(-1, 1)
        )

        # Prepare dataset
        dataset = self.SalesDataset(scaled_data, look_back)

        # Split dataset
        split = int(len(dataset) * 0.8)
        train_data = dataset[:split]
        test_data = dataset[split:]

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # Define the LSTM model
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])
                return out

        # Model and hyperparameters
        input_dim = 1
        hidden_dim = 50
        output_dim = 1
        num_layers = 2
        model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        # Training
        self.logger.info(f"Training PyTorch LSTM model, {self.model}...")
        epochs = 10
        train_losses = []  # To store training loss for plotting

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                # Unsqueeze target for shape compatibility
                loss = criterion(outputs, targets.unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_loss = train_loss / len(train_loader)
            train_losses.append(avg_loss)
            self.logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.logger.info(f"PyTorch LSTM model, {self.model} training complete.")

        # Evaluate on test data
        self.logger.info(f"Evaluating model, {self.model} on test data...")
        model.eval()
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                test_predictions.append(outputs.squeeze(1).numpy())
                test_targets.append(targets.numpy())

        # Flatten predictions and targets
        test_predictions = np.concatenate(test_predictions)
        test_targets = np.concatenate(test_targets)

        # Rescale to original scale
        test_predictions = self.scalar.inverse_transform(
            test_predictions.reshape(-1, 1)
        ).flatten()
        test_targets = self.scalar.inverse_transform(
            test_targets.reshape(-1, 1)
        ).flatten()

        # Calculate metrics
        rmse = np.sqrt(np.mean((test_predictions - test_targets) ** 2))
        mae = np.mean(np.abs(test_predictions - test_targets))
        self.logger.info(f"Test RMSE: {rmse:.4f}")
        self.logger.info(f"Test MAE: {mae:.4f}")

        # Plot training loss
        # import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig("plots/training_loss.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Plot predictions vs targets
        plt.figure(figsize=(10, 6))
        plt.plot(test_targets, label="True Values")
        plt.plot(test_predictions, label="Predictions")
        plt.title("Test Predictions vs True Values")
        plt.xlabel("Time Step")
        plt.ylabel("Sales")
        plt.legend()
        plt.grid(True)
        plt.savefig("plots/predictions_vs_targets.png", dpi=300, bbox_inches="tight")
        plt.show()

        return model
