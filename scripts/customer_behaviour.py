import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

import logging


class CustomerBehaviourEDA:
  def __init__(self, data):
    self.data = data

    logging.basicConfig(
            filename="customer_behavior.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    logging.info("CustomerBehaviorEDA instance created.")


