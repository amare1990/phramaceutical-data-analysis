""" Building ML and DL Models module for store sales prediction. """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

import datetime
import pickle


import logging


class StoreSalesPrediction:
  """ A store sales prediction class"""
  def __init__(self, train_data: pd.DataFrame):
    self.data = train_data
    # self.test_data = test_data

    self.model = None
    self.scalar = StandardScaler()
    self.logger = self.setup_logger()


  @staticmethod
  def setup_logger():
    logger = logging.getLogger("StoreSalesPrediction")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger



  def preprocess_data(self):
    """ Preprocess train and test data. """
    self.logger.info("Starting preprocessing...")

    # Strip white spaces in column names if there
    dataset = self.data
    dataset.columns = dataset.columns.str.strip()

    # Scale numerical columns (exclude 'Id' and 'Sales' if present)
    numeric_cols = dataset.select_dtypes(include=np.number).columns

    # Handle datetime columns
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset['Weekday'] = dataset['Date'].dt.weekday
    dataset['IsWeekend'] = dataset['Weekday'] >= 5
    dataset['Month'] = dataset['Date'].dt.month
    dataset['IsBeginningOfMonth'] = dataset['Date'].dt.day <= 10
    dataset['IsMidMonth'] = (dataset['Date'].dt.day > 10) & (dataset['Date'].dt.day <= 20)
    dataset['IsEndOfMonth'] = dataset['Date'].dt.day > 20
    dataset['Year'] = dataset['Date'].dt.year

    # One-hot encode the month segment features
    month_segment_encoded = pd.get_dummies(dataset[['IsBeginningOfMonth', 'IsMidMonth', 'IsEndOfMonth']], drop_first=True)
    dataset = pd.concat([dataset, month_segment_encoded], axis=1)

    current_year = dataset['Date'].dt.year
    current_week = dataset['Date'].dt.isocalendar().week

    # Calculate elapsed years and weeks for Promo2
    dataset['Promo2ElapsedYears'] = current_year - dataset['Promo2SinceYear']
    dataset['Promo2ElapsedWeeks'] = current_week - dataset['Promo2SinceWeek']
    dataset['StoreOpenMonths'] = (
        (dataset['Date'].dt.year - dataset['CompetitionOpenSinceYear']) * 12 +
        (dataset['Date'].dt.month - dataset['CompetitionOpenSinceMonth'])
    )
    dataset['IsPublicHoliday'] = dataset['StateHoliday'].isin(['a', 'b', 'c']).astype(int)

    # One-hot encode StoreType and Assortment
    store_type_encoded = pd.get_dummies(dataset['StoreType'], prefix='StoreType')
    assortment_encoded = pd.get_dummies(dataset['Assortment'], prefix='Assortment')
    dataset = pd.concat([dataset, store_type_encoded, assortment_encoded], axis=1)
    dataset.drop(columns=['StoreType', 'Assortment'], inplace=True)

    # Handle PromoInterval (One-Hot Encoding, including 'None' for non-participating stores)
    if 'PromoInterval' in dataset.columns:
        dataset['Promo2_Participation'] = (dataset['PromoInterval'] != 'None').astype(int)
        promo_interval_encoded = pd.get_dummies(dataset['PromoInterval'], prefix='PromoInterval')
        dataset = pd.concat([dataset, promo_interval_encoded], axis=1)
        dataset.drop(columns=['PromoInterval'], inplace=True)
    else:
        self.logger.warning("'PromoInterval' column is missing!")
        dataset['Promo2_Participation'] = 0  # Default value
        dataset.drop(columns=['PromoInterval'], inplace=True, errors='ignore')

    # Handle StateHoliday (One-Hot Encoding)
    stateholiday_encoded = pd.get_dummies(dataset['StateHoliday'], prefix='StateHoliday', drop_first=False)
    dataset = pd.concat([dataset, stateholiday_encoded], axis=1)
    dataset.drop(columns=['StateHoliday'], inplace=True, errors='ignore')

    # Assign self.data to dataset
    self.data = dataset

    self.logger.info("Preprocessing completed.")

    return self.data


  def build_sklearn_model(self):
      self.logger.info("Building sklearn model...")

      # Define the feature matrix (X) and target (y) using preprocessed data
      columns_to_drop = ['Date']
      X = self.data.drop(columns=['Sales'] + columns_to_drop, axis=1, errors='ignore')  # Drop 'Sales' and date-like features
      y = self.data['Sales']


      # Split self.data into train and test data
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

      # Define the feature matrix (X) and target (y) using preprocessed data
      # X_train = X_train.drop(['Sales', 'Date', 'Id'], axis=1)
      # y_train = y_train['Sales']

      # X_test = X_test.drop(['Sales', 'Date', 'Id'], axis=1)
      # y_test = y_test['Sales']

      # Check if the columns in train and test data are aligned
      assert all(X_train.columns == X_test.columns), "Feature columns in train and test data do not match!"

      # Define the pipeline with a RandomForestRegressor
      pipeline = Pipeline([
          ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
      ])

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
        model = self.model.named_steps['regressor']
        importance = model.feature_importances_
        features = self.data.drop(['Sales', 'Date'], axis=1).columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
        importance_df.sort_values(by='Importance', ascending=False, inplace=True)


        self.logger.info('Feature Importance: ')
        self.logger.info(importance_df)


        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

  def save_model(self):
     timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
     filename = f"Store_sales_model_{timestamp}.pkl"

     with open(filename, "wb") as file:
        pickle.dump(self.model, file)

     self.logger.info(f'Model saved as {filename}. ')

