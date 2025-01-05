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
    self.logger = self.set


  @staticmethod
  def setup_logger():
    logger = logging.getLogger("StoreSalesPrediction")
    logger.setLevel(logging.INFO)
    handler = logger.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

