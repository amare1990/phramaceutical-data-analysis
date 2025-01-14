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

  def preprocess_input(self, input_data):
        """
        Preprocess input data for prediction.
        """
        try:
            # Example preprocessing (adapt to your dataset and model requirements)
            input_array = np.array(input_data).reshape(1, -1)
            return input_array
        except Exception as e:
            raise ValueError(f"Error preprocessing input: {e}")


  def predict(self, input_data):
    """
    Preprocess input data and make predictions.
    :param data: test_data
    :return: model predictions
    """
    preprocessed_data = self.preprocess_input(input_data)
    predictions = self.model.predict(preprocessed_data)
    return predictions.tolist()


