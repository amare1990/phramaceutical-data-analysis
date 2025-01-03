import pandas as pd
import numpy as np


from scripts.customer_behaviour import CustomerBehaviourEDA
store_path = "../data/store.csv"
train_path = "../data/train.csv"
test_path = "../data/test.csv"
customer_eda = CustomerBehaviourEDA(store_path, train_path, test_path)


def customer_behaviour_analysis_pipline_processor():
  customer_eda.load_data()
  print(f"The shape of the original store data is {customer_eda.store_data.shape}")
  print(f"The shape of the original train data is {customer_eda.train_data.shape}")
  print(f"The shape of the original test data is {customer_eda.test_data.shape}")
  print("\n\n************************************************************************************")
  print("Store , train and test date columns before merging")
  print(f"Store data features\n {customer_eda.store_data.columns}")
  print(f"Train data features\n {customer_eda.train_data.columns}")
  print(f"test data features\n {customer_eda.test_data.columns}")
  print("\n\n************************************************************************************")
  print("Fixing mixed data types")

  # Convert the entire 7th column to strings
  print(customer_eda.train_data.iloc[:, 7].unique())
  customer_eda.train_data.iloc[:, 7] = customer_eda.train_data.iloc[:, 7].astype(str)
  # Verify the unique values after fixing mixed data
  print(customer_eda.train_data.iloc[:, 7].unique())

  print("\n\n************************************************************************************")
  print("Merging data: store data with train data, store data with test data")
  customer_eda.merge_datasets()
  print(f"The shape of the merged train data is {customer_eda.train_data.shape}")
  print(f"The shape of the merged test data is {customer_eda.test_data.shape}")
  print("\n\n************************************************************************************")
  print("train and test date columns after merging")
  print(f"train data features\n {customer_eda.train_data.columns}")
  print(f"test data features\n {customer_eda.test_data.columns}")

  print("\n\n************************************************************************************")
  print("Cleaning train data and saving the cleaned train data")
  train_cleaned_path = "../data/train_cleaned_data.csv"
  test_cleaned_path = "../data/test_cleaned_data.csv"
  customer_eda.clean_data(train_cleaned_path, test_cleaned_path)

  print(f"The shape of the cleaned train data is {customer_eda.train_data.shape}")
  print(f"Cleaned train data features\n {customer_eda.train_data.columns}")

  print("\n\n************************************************************************************")
  print("Visualizations of Sales and customers distribution via histograms")
  customer_eda.visualize_distributions()

  print("\n\n************************************************************************************")
  print("Visualizations for avergae sales and average customers per promotions")
  # Visualizations for average sales with promotions
  customer_eda.analyze_promotions()

  print("\n\n************************************************************************************")
  print("Visualizations for the effect of state holidayas over sales and customers")
  customer_eda.explore_holiday_effects()

  print("\n\n************************************************************************************")
  print("Heatmap for relationships between numerical features")
  customer_eda.check_correlations()

  print("\n\n************************************************************************************")
  print("Statistical summary for both numerical and non-numerical features")
  customer_eda.statistical_summary()


  print("\n\n************************************************************************************")
  print("Top ten stores promo is required")
  customer_eda.analyze_promo_effectiveness()

  print("\n\n************************************************************************************")
  print("Trends of customer behavior during store opening and closing times")
  customer_eda.analyze_customer_behavior_opening_hours

  print("\n\n************************************************************************************")
  print("Stores that are open in weekdayas and the effect of this over sales")
  customer_eda.analyze_weekday_opening_effect()

  print("\n\n************************************************************************************")
  print("How assortment types affect sales")
  customer_eda.analyze_assortment_impact()


  print("\n\n************************************************************************************")
  print("Effect of the distance to the next competitor on sales")
  customer_eda.analyze_competitor_distance_impact()

  print("\n\n************************************************************************************")
  print("Effect of opening or re-opening of new competitor on stores")
  customer_eda.analyze_new_competitor_impact




