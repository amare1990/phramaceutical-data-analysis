import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

        # Assign merged_train_data and merged_test_data to the class instance variables
        self.train_data = self.merged_train_data
        self.test_data = self.merged_test_data

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

    # self.merged_train_data, self.merged_test_data = self.get_merged_data()

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
    # for dataset_name, dataset in {"train_data": self.merged_train_data, "store_data": self.merged_test_data}.items():
    if 'CompetitionDistance' in self.train_data.columns:
        self.train_data['CompetitionDistance'].fillna(self.train_data['CompetitionDistance'].median(), inplace=True)
        logging.info(f"Filled missing 'CompetitionDistance' in train data.")

    if 'PromoInterval' in self.train_data.columns:
        self.train_data['PromoInterval'].fillna('None', inplace=True)
        logging.info(f"Filled missing 'PromoInterval' in train data.")

    # Handle outliers in numeric columns for train_data and store_data
    # for dataset_name, dataset in {"train_data": self.merged_train_data, "store_data": self.merged_test_data}.items():
    numeric_columns = ['Sales', 'Customers', 'CompetitionDistance']
    for col in numeric_columns:
        if col in self.train_data.columns:
            q1 = self.train_data[col].quantile(0.25)
            q3 = self.train_data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = self.train_data[(self.train_data[col] < lower_bound) | (self.train_data[col] > upper_bound)]
            self.train_data.drop(outliers.index, inplace=True)
            logging.info(f"Outliers removed from column '{col}' in train_data. Total removed: {len(outliers)}")

    # Save the cleaned data, i.e., the train data
    self.train_data.to_csv(train_cleaned_path, index=False)
    # self.test_data.to_csv(test_cleaned_path, index=False)
    logging.info(f"Cleaned data saved to {train_cleaned_path}.")


  def visualize_distributions(self):
        """Visualize distributions of key features."""
        logging.info("Visualizing feature distributions.")

        plt.figure(figsize=(10, 5))
        sns.histplot(self.train_data['Sales'], kde=True, bins=30)
        plt.title('Sales Distribution')
        plt.show()
        plt.savefig(
                f"plots/sales_distribution.png",
                dpi=300,
                bbox_inches='tight')
        logging.info("Sales distribution plotted.")

        plt.figure(figsize=(10, 5))
        sns.histplot(self.train_data['Customers'], kde=True, bins=30)
        plt.title('Customers Distribution')
        plt.show()
        plt.savefig(
                f"plots/customers_distribution.png",
                dpi=300,
                bbox_inches='tight')
        logging.info("Customers distribution plotted.")

  def analyze_promotions(self):
    """Analyze the impact of promotions on sales and customers."""
    logging.info("Analyzing promotion effects on sales and customers.")

    # Calculate average sales and customers for promo and no-promo
    promo_sales = self.train_data.groupby('Promo')['Sales'].mean()
    promo_customers = self.train_data.groupby('Promo')['Customers'].mean()

    # Log the calculated values
    logging.info(f"Average sales with promo: {promo_sales[1]:.2f}, without promo: {promo_sales[0]:.2f}.")
    logging.info(f"Average customers with promo: {promo_customers[1]:.2f}, without promo: {promo_customers[0]:.2f}.")

    # Plot average sales by promo status
    plt.figure(figsize=(8, 5))
    promo_sales.plot(kind='bar', color=['blue', 'orange'], alpha=0.7)
    plt.title('Average Sales by Promo Status')
    plt.xlabel('Promo Status (0 = No Promo, 1 = Promo)')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=0)
    for i, v in enumerate(promo_sales):
        plt.text(i, v + 0.02 * max(promo_sales), f"{v:.1f}", ha='center', fontsize=10)
    plt.show()
    plt.savefig(
                f"plots/average_sales_per_promo.png",
                dpi=300,
                bbox_inches='tight')
    logging.info("Plotted average sales by promo status.")

    # Plot average customers by promo status
    plt.figure(figsize=(8, 5))
    promo_customers.plot(kind='bar', color=['blue', 'orange'], alpha=0.7)
    plt.title('Average Customers by Promo Status')
    plt.xlabel('Promo Status (0 = No Promo, 1 = Promo)')
    plt.ylabel('Average Customers')
    plt.xticks(rotation=0)
    for i, v in enumerate(promo_customers):
        plt.text(i, v + 0.02 * max(promo_customers), f"{v:.1f}", ha='center', fontsize=10)
    plt.show()
    plt.savefig(
                f"plots/average customers per promo.png",
                dpi=300,
                bbox_inches='tight')
    logging.info("Plotted average customers by promo status.")

  def explore_holiday_effects(self):
        """Analyze sales behavior around holidays."""
        logging.info("Exploring holiday effects on sales.")

        holiday_sales = self.train_data.groupby('StateHoliday')['Sales'].mean()

        holiday_sales.plot(kind='bar', title='Average Sales by State Holiday', color=['blue', 'green', 'red', 'orange'])
        plt.show()
        plt.savefig(
                f"plots/average sales by StateHoliday.png",
                dpi=300,
                bbox_inches='tight')
        logging.info("Plotted average sales by state holiday.")

        logging.info("Exploring holiday effects on Customers.")
        holiday_customers = self.train_data.groupby('StateHoliday')['Customers'].mean()
        holiday_customers.plot(kind='bar', title='Average Customers by State Holiday', color=['blue', 'green', 'red', 'orange'])
        plt.show()
        plt.savefig(
                f"plots/average customers by StateHoliday.png",
                dpi=300,
                bbox_inches='tight')
        logging.info("Plotted average Customers by state holiday.")


  def check_correlations(self):
      """Check correlations between numerical features. """
      logging.info("Checking correlations between numerical features. ")

      # Select only numerical columns
      numerical_data = self.train_data.select_dtypes(include=['number'])

      # Compute the correlation matrix
      correlations = numerical_data.corr()

      plt.figure(figsize=(12, 8))
      sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt='.2f')
      plt.title("Feature Correlation Matrix")
      plt.show()
      plt.savefig(
                f"plots/feature correlation matrix.png",
                dpi=300,
                bbox_inches='tight')
      logging.info("Plotting correlation matrix done. ")

  def statistical_summary(self):
    """Generate and log summary statistics for both numerical and categorical data."""
    logging.info("Generating summary statistics for training data.")

    # Generate summary for numerical data
    num_summary = self.train_data.describe()

    # Generate summary for categorical (object) data
    cat_summary = self.train_data.describe(include=['object'])

    logging.info("Summary statistics for numerical data:")
    logging.info(f"\n{num_summary}")

    logging.info("Summary statistics for categorical data:")
    logging.info(f"\n{cat_summary}")

    print("Numerical Summary:")
    print(num_summary)
    print("\nCategorical Summary:")
    print(cat_summary)


  def analyze_promo_effectiveness(self):
    """Analyze the effectiveness of promotions and recommend deployment strategies."""
    logging.info("Analyzing promotion effectiveness.")

    # Group by Store and Promo to analyze average sales and customers
    promo_effectiveness = self.train_data.groupby(['Store', 'Promo'])[['Sales', 'Customers']].mean().reset_index()

    # Identify stores with the highest sales uplift during promotions
    promo_uplift = promo_effectiveness.pivot(index='Store', columns='Promo', values='Sales')
    promo_uplift['Uplift'] = promo_uplift[1] - promo_uplift[0]
    recommended_stores = promo_uplift.sort_values(by='Uplift', ascending=False).head(10)

    logging.info(f"Top stores for promo deployment:\n{recommended_stores}")
    print(f"Top stores for promo deployment:\n{recommended_stores}")


  def analyze_customer_behavior_opening_hours(self):
    """Analyze customer behavior during store opening and closing times."""
    logging.info("Analyzing customer behavior during store opening and closing times.")

    # Filter data for open stores
    open_stores = self.train_data[self.train_data['Open'] == 1]

    # Group by day of the week
    customer_trends = open_stores.groupby('DayOfWeek')['Customers'].mean()

    # Plot customer trends by day of the week
    customer_trends.plot(kind='line', marker='o', title='Customer Behavior by Day of Week')
    plt.ylabel('Average Customers')
    plt.xlabel('Day of Week')
    plt.show()
    logging.info("Plotted customer behavior during store opening and closing times.")

  def analyze_weekday_opening_effect(self):
    """Analyze stores open on all weekdays and their weekend sales."""
    logging.info("Identifying stores open on all weekdays.")

    # Check if stores are open on all weekdays
    weekday_data = self.train_data[self.train_data['DayOfWeek'] < 6]
    all_weekdays_open = weekday_data.groupby('Store')['Open'].sum() == 5

    # Filter stores open all weekdays
    open_all_weekdays = all_weekdays_open[all_weekdays_open].index
    weekend_data = self.train_data[(self.train_data['Store'].isin(open_all_weekdays)) & (self.train_data['DayOfWeek'] >= 6)]

    # Compare sales on weekends
    weekend_sales = weekend_data.groupby('DayOfWeek')['Sales'].mean()
    weekend_sales.plot(kind='bar', title='Weekend Sales for Stores Open All Weekdays')
    plt.ylabel('Average Sales')
    plt.show()

  def analyze_assortment_impact(self):
    """Analyze the impact of assortment type on sales."""
    logging.info("Analyzing how assortment type affects sales.")

    # Group by assortment and calculate average sales
    assortment_sales = self.train_data.groupby('Assortment')['Sales'].mean()

    # Plot results
    assortment_sales.plot(kind='bar', title='Average Sales by Assortment Type', color=['blue', 'orange', 'green'])
    plt.ylabel('Average Sales')
    plt.show()

  def analyze_competitor_distance_impact(self):
    """Analyze the impact of competitor distance on sales."""
    logging.info("Analyzing the impact of competitor distance on sales.")

    # Bin competitor distances
    self.train_data['CompDistanceBin'] = pd.cut(
        self.train_data['CompetitionDistance'],
        bins=[0, 1000, 5000, 10000, float('inf')],
        labels=['<1km', '1-5km', '5-10km', '>10km']
    )

    # Group by distance bins and calculate sales
    distance_impact = self.train_data.groupby('CompDistanceBin')['Sales'].mean()

    # Plot results
    distance_impact.plot(kind='bar', title='Average Sales by Competitor Distance')
    plt.ylabel('Average Sales')
    plt.xlabel('Competitor Distance')
    plt.show()

  def analyze_new_competitor_impact(self):
    """Analyze the impact of new competitor openings on sales."""
    logging.info("Analyzing new competitor impact on sales.")

    # Filter stores with missing competitor distance initially
    na_comp_stores = self.train_data[self.train_data['CompetitionDistance'].isna()]

    # Later data where competitor distance becomes available
    available_comp_stores = self.train_data[~self.train_data['CompetitionDistance'].isna()]

    # Compare average sales
    avg_sales_na = na_comp_stores['Sales'].mean()
    avg_sales_available = available_comp_stores['Sales'].mean()

    logging.info(f"Average sales before competitor data: {avg_sales_na}")
    logging.info(f"Average sales after competitor data: {avg_sales_available}")

    # Bar plot comparison
    pd.DataFrame({
        'Before': [avg_sales_na],
        'After': [avg_sales_available]
    }).plot(kind='bar', title='Impact of New Competitor on Sales', color=['red', 'green'])
    plt.ylabel('Average Sales')
    plt.show()




  def analyze_new_competitor_effect(self):
    """
    Analyze the effect of new competitors opening or reopening on stores.
    Focuses on stores with NA as CompetitionDistance that later have values.
    """
    logging.info("Analyzing the effect of new competitors opening or reopening.")

    # Identify stores with missing CompetitionDistance initially
    stores_with_na = self.train_data[self.train_data['CompetitionDistance'].isna()]['Store'].unique()

    # Filter data for these stores
    na_data = self.train_data[self.train_data['Store'].isin(stores_with_na)]

    # Separate data into before and after competitor information is available
    na_before = na_data[na_data['CompetitionDistance'].isna()]
    na_after = na_data[~na_data['CompetitionDistance'].isna()]

    # Group by store and calculate average sales before and after
    avg_sales_before = na_before.groupby('Store')['Sales'].mean()
    avg_sales_after = na_after.groupby('Store')['Sales'].mean()

    # Merge the data for comparison
    sales_comparison = pd.DataFrame({
        'Before': avg_sales_before,
        'After': avg_sales_after
    }).dropna()

    # Add a column to calculate the sales difference
    sales_comparison['Difference'] = sales_comparison['After'] - sales_comparison['Before']

    # Output the result
    logging.info(f"Sales comparison before and after competitor information:\n{sales_comparison}")
    print(sales_comparison)

    # Plot the differences
    sales_comparison['Difference'].plot(kind='bar', figsize=(12, 6), title='Sales Difference Before and After Competitor Opening')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.ylabel('Sales Difference')
    plt.xlabel('Store')
    plt.show()


