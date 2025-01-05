""" Building ML and DL Models module for store sales prediction. """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler


import logging


class StoreSalesPrediction:
  """ A store sales prediction class"""
  def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
    self.train_data = train_data
    self.test_data = test_data

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

    def preprocess_single_dataset(dataset, is_train=True):
        # Strip white spaces in column names if there
        dataset.columns = dataset.columns.str.strip()

        # Scale numerical columns (exclude 'Id' and 'Sales' if present)
        numeric_cols = dataset.select_dtypes(include=np.number).columns
        # Exclude 'Sales' and 'Customers' columns during testing
        exclude_cols = ['Id']
        if 'Sales' in numeric_cols or 'Customers' in numeric_cols and not is_train:
            exclude_cols.append('Sales')
            exclude_cols.append('Customers')
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]


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

        return dataset

    # Preprocess train and test datasets
    self.train_data = preprocess_single_dataset(self.train_data, is_train=True)
    self.test_data = preprocess_single_dataset(self.test_data, is_train=False)

    self.logger.info("Preprocessing completed.")
