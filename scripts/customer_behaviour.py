import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

import logging


class CustomerBehaviourEDA:
  def __init__(self, store_path, train_path, test_path):
    self.store_path = store_path
    self.train_path = train_path
    self.test_path = test_path
    self.store_data = None
    self.train_data = None
    self.test_data = None
    self.merged_train_data = None
    self.merged_test_data = None

    logging.basicConfig(
            filename="customer_behavior.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    logging.info("CustomerBehaviorEDA instance created.")


  def load_data(self):
        """
        Load the datasets from the provided file paths.
        """
        self.store_data = pd.read_csv(self.store_path)
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)



  def save_data(self, saving_data_path):
      """

      Save dataset into the provided path
      """
      pd.to_csv(saving_data_path)


  def merge_datasets(self):
        """
        Merge the store, train, and test datasets on the 'Store' column.
        """
        # Check if data is loaded
        if self.store_data is None or self.train_data is None or self.test_data is None:
            raise ValueError("Datasets are not loaded. Call the `load_data` method first.")

        # Merge store with train
        self.merged_train_data = pd.merge(self.train_data, self.store_data, on='Store', how='inner')

        # Merge store with test
        self.merged_test_data = pd.merge(self.test_data, self.store_data, on='Store', how='inner')

  def get_merged_data(self):
        """
        Return the merged train and test datasets.
        """
        if self.merged_train_data is None or self.merged_test_data is None:
            raise ValueError("Datasets are not merged. Call the `merge_datasets` method first.")

        return self.merged_train_data, self.merged_test_data


  def clean_data(self, train_cleaned_path, test_cleaned_path):
    """Clean the data by fixing mixed data types, handling missing values, and removing outliers."""
    logging.info("Starting data cleaning.")

    self.merged_train_data, self.merged_test_data = self.get_merged_data()

    # Detect and fix mixed data types for all columns in train_data and store_data
    # for dataset_name, dataset in {"train_data": self.merged_train_data, "store_data": self.merged_test_data}.items():
    #     for col in dataset.columns:
    #         unique_types = set(type(val) for val in dataset[col].dropna())
    #         if len(unique_types) > 1:
    #             logging.warning(f"Column '{col}' in {dataset_name} has mixed data types: {unique_types}. Analyzing suitable type.")
    #             # Determine appropriate type: prefer numeric if possible, otherwise string
    #             try:
    #                 dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
    #                 if dataset[col].isnull().sum() == 0:  # Conversion successful without data loss
    #                     logging.info(f"Column '{col}' in {dataset_name} converted to numeric.")
    #                 else:
    #                     dataset[col] = dataset[col].astype(str)
    #                     logging.info(f"Column '{col}' in {dataset_name} converted to string due to data loss.")
    #             except Exception:
    #                 dataset[col] = dataset[col].astype(str)
    #                 logging.info(f"Column '{col}' in {dataset_name} converted to string.")

    # Fill missing values in train_data and store_data
    for dataset_name, dataset in {"train_data": self.merged_train_data, "store_data": self.merged_test_data}.items():
        if 'CompetitionDistance' in dataset.columns:
            dataset['CompetitionDistance'].fillna(dataset['CompetitionDistance'].median(), inplace=True)
            logging.info(f"Filled missing 'CompetitionDistance' in {dataset_name}.")

        if 'PromoInterval' in dataset.columns:
            dataset['PromoInterval'].fillna('None', inplace=True)
            logging.info(f"Filled missing 'PromoInterval' in {dataset_name}.")

    # Handle outliers in numeric columns for train_data and store_data
    # for dataset_name, dataset in {"train_data": self.merged_train_data, "store_data": self.merged_test_data}.items():
    numeric_columns = ['Sales', 'Customers', 'CompetitionDistance']
    for col in numeric_columns:
        if col in dataset.columns:
            q1 = dataset[col].quantile(0.25)
            q3 = dataset[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = dataset[(dataset[col] < lower_bound) | (dataset[col] > upper_bound)]
            dataset.drop(outliers.index, inplace=True)
            logging.info(f"Outliers removed from column '{col}' in {dataset_name}. Total removed: {len(outliers)}")


    # Save the cleaned data
    self.merged_train_data.to_csv(train_cleaned_path, index=False)
    self.merged_test_data.to_csv(test_cleaned_path, index=False)
    logging.info(f"Cleaned data saved to {train_cleaned_path} and {test_cleaned_path}.")








