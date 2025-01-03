import pandas as pd
import numpy as np


from scripts.customer_behaviour import CustomerBehaviourEDA
store_path = "../data/store.csv"
train_path = "../data/train.csv"
test_path = "../data/test.csv"
customer_eda = CustomerBehaviourEDA(store_path, train_path, test_path)


def customer_behaviour_analysis_pipline_processor():
  customer_eda.load_data()
  print(f"\nThe shape of the original store data is {customer_eda.store_data.shape}\n")
  print(f"The shape of the original train data is {customer_eda.train_data.shape}\n")
  print(f"The shape of the original test data is {customer_eda.test_data.shape}\n")
  print("\n\n************************************************************************************")
  print("Store , train and test date columns before merging\n")
  print(f"Store data features\n\n {customer_eda.store_data.columns}\n")
  print(f"Train data features\n\n {customer_eda.train_data.columns}\n")
  print(f"test data features\n\n {customer_eda.test_data.columns}\n")
  print("\n\n************************************************************************************")
  print("Fixing mixed data types problems\n")

  # Convert the entire 7th column to strings
  print(f"Unique values for the 7th column of the train data before fixing {customer_eda.train_data.iloc[:, 7].unique()}")
  customer_eda.train_data.iloc[:, 7] = customer_eda.train_data.iloc[:, 7].astype(str)
  # Verify the unique values after fixing mixed data
  print(f"Unique values for the 7th column of the train data afetr fixing {customer_eda.train_data.iloc[:, 7].unique()}")

  print("\n\n************************************************************************************")
  print("Merging data: store data with train data, store data with test data\n")
  customer_eda.merge_datasets()
  print(f"The shape of the merged train data is {customer_eda.train_data.shape}")
  print(f"The shape of the merged test data is {customer_eda.test_data.shape}")
  print("\n\n************************************************************************************")
  print("train and test date columns after merging\n")
  print(f"train data features\n {customer_eda.train_data.columns}")
  print(f"test data features\n {customer_eda.test_data.columns}")

  print("\n\n************************************************************************************")
  print("Cleaning train data and saving the cleaned train data\n")
  train_cleaned_path = "../data/train_cleaned_data.csv"
  test_cleaned_path = "../data/test_cleaned_data.csv"
  customer_eda.clean_data(train_cleaned_path, test_cleaned_path)

  print(f"\nThe shape of the cleaned train data is {customer_eda.train_data.shape}")
  print(f"Cleaned train data features\n {customer_eda.train_data.columns}")

  print("\n\n************************************************************************************")
  print("Visualizations of Sales and customers distribution via histograms\n")
  customer_eda.visualize_distributions()

  print("\n\n************************************************************************************")
  print("Visualizations for avergae sales and average customers per promotions\n")
  # Visualizations for average sales with promotions
  customer_eda.analyze_promotions()

  print("\n\n************************************************************************************")
  print("Visualizations for the effect of state holidayas over sales and customers\n")
  customer_eda.explore_holiday_effects()

  print("\n\n************************************************************************************")
  print("Heatmap for relationships between numerical features\n")
  customer_eda.check_correlations()

  print("\n\n************************************************************************************")
  print("Statistical summary for both numerical and non-numerical features\n")
  customer_eda.statistical_summary()


  print("\n\n************************************************************************************")
  print("Top ten stores promo should be deployed for\n\n")
  customer_eda.analyze_promo_effectiveness()

  print("\n\n************************************************************************************")
  print("Trends of customer behavior during store opening and closing times\n")
  customer_eda.analyze_customer_behavior_opening_hours

  print("\n\n************************************************************************************")
  print("Stores that are open in weekdayas and the effect of this over sales\n")
  customer_eda.analyze_weekday_opening_effect()

  print("\n\n************************************************************************************")
  print("How assortment types affect sales\n")
  customer_eda.analyze_assortment_impact()


  print("\n\n************************************************************************************")
  print("Effect of the distance to the next competitor on sales\n")
  customer_eda.analyze_competitor_distance_impact()

  print("\n\n************************************************************************************")
  print("Effect of opening or re-opening of new competitor on stores\n")
  customer_eda.analyze_new_competitor_impact()




