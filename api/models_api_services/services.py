"""ApI Service Implementation. """

import pickle
import numpy as np


class PredictionService:
  def __init__(self, model_path='../../data/model.pkl'):
    self.model = pickle.load(model_path)


def predict(self, data):
  """

  Preprocess input data and make predictions.
  :param data: test_data
  :return: model predictions
  """

  input_data = np.array(data)
  predictions = self.model.predict(input_data)
  return predictions.tolist()


