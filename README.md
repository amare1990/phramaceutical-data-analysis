# Car Insurance Risk Assessment and Predictive Analysis

> Car Insurance Risk Assessment and Predictive Analysis is a data science project designed to statistically analyze and extract insights from the data from AlphaCare Insurance Solutions (ACIS), a car insurance company in South Africa. This data science project attempts to conduct Exploratory Data Discovery and Analysis (EDA). It finds key insights from the data using statistical and EDA analysis and from the histogram, bar charts, and correlation matrix. It is implemented using Python programming language and various tools including jupyter notebook.

## Built With

- Major languages used: Python3
- Libraries: numpy, pandas, seaborn, matplotlib.pyplot, scikit-learn
- Tools and Technlogies used: jupyter notebook, Git, GitHub, Gitflow, VS code editor.

## Demonstration and Website

[Deployment link](Soon!)

## Getting Started

You can clone my project and use it freely and then contribute to this project.

- Get the local copy, by running `git clone https://github.com/amare1990/phramaceutical-data-analysis.git` command in the directory of your local machine.
- Go to the repo main directory, run `cd phramaceutical-data-analysis` command
- Create python environment by running `python3 -m venv venm-name`, where `ven-name` is your python environment you create
- Activate it by running:
- `source venv-name/bin/activate` on linux os command prompt if you use linux os
- `myenv\Scripts\activate` on windows os command prompt if you use windows os.

- After that you have to install all the necessary Python libraries and tools by running `pip install -r requirements.txt`
- To run this project, run `jupyter notebook` command from the main directory of the repo

### Prerequisites

- You have to install Python (version 3.8.10 minimum), pip, git, vscode.

### Dataset

 - `store.csv`, `train.csv`, and `test.csv` are the datasets from `Kaggle` used to conduct statistical Exploratory Data Analysis (EDA).
 - Run `df = pandas.read_csv("/path to your dataset")` to get the pandas data =frame for each dataset.
 - The store dataset is merged with the train and test data so that the `store` feature both in the train and test data are expanded more with details.

### Project Requirements
- Git, GitHub setup, adding `pylint' in the GitHub workflows
- Statistical and EDA analysis on the data, ploting
- Gaining insightful information by cinducting various tricks and visualizations

#### GitHub Action
- Go to the main directory of this repo, create paths, `.github/workflows`. And then add `pylint` linters
- Make it to check when Pull request is created
- Run `pylint scripts/script_name.py` to check if the code follows the standard format
- Run `autopep8 --in-place --aggressive --aggressive scripts/script_name.py` to automatically fix some linters errors

### Customer Purchasing Behaviour Analysis

The main functionality is implemented in `customer_behaviour.py` module and the `customer_behaviour_pipeline_processing.py` pipelines all processes. `customer_behavior_eda.ipynb` is the notebook you call the method in the pipeline processor module.
In this portion of the task, the following analysis has been conducted.

- Fix mixed data types
  Converted into string data type

- Data Summary:
    Summarize statistical descriptive statistics for both numerical features and object type features too.

- Data Quality Check:
    Identify and address missing values.
    Detect outliers and remove.
    Replace nan/empty/infinity/-infinity with median.
    Save cleaned data
- Visualizations of Sales and customers distribution via histograms
- Visualizations for the effect of state holidayas over sales and customers.
- Heatmap for relationships between numerical features
- Finding top ten stores promo should be deployed for
- Trends of customer behavior during store opening and closing times
- Viewing stores that are open in weekdayas and the effect of this over sales
- viewing how assortment types affect sales via visualizations
- Effect of the distance to the next competitor on sales via visualizations
- Effect of opening or re-opening of new competitor on stores via visualizations

  Run /open the jupyter notebook named `customer_behavior_eda.ipynb` to clean data




> #### You can gain more insights by running the jupter notebook and view plots.


### More information
- You can refer to [this link]() to gain more insights about the reports of this project results.

## Authors

👤 **Amare Kassa**

- GitHub: [@githubhandle](https://github.com/amare1990)
- Twitter: [@twitterhandle](https://twitter.com/@amaremek)
- LinkedIn: [@linkedInHandle](https://www.linkedin.com/in/amaremek/)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

Feel free to check the [issues page](https://github.com/amare1990/phramaceutical-data-analysis/issues).

## Show your support

Give a ⭐️ if you like this project, and you are welcome to contribute to this project!

## Acknowledgments

- Hat tip to anyone whose code was referenced to.
- Thanks to the 10 academy and Kifiya financial instituion that gives me an opportunity to do this project

## 📝 License

This project is [MIT](./LICENSE) licensed.
