"""ApI Service Implementation. """

import pickle
import numpy as np


class PredictionService:
  def __init__(self, model_path='../../data/model.pkl'):
    # self.model = pickle.load(model_path)
    self.model = self.load_model(model_path)


  def load_model(model_path):
    try:
      with open(model_path, 'rb') as file:
        model = pickle.load(file)
      return model
    except Exception as e:
      raise ValueError(f"Error loading model: {e}")



def predict(self, data):
  """

  Preprocess input data and make predictions.
  :param data: test_data
  :return: model predictions
  """

  input_data = np.array(data)
  predictions = self.model.predict(input_data)
  return predictions.tolist()


