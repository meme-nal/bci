import numpy
import typing

import datasets
import processing
import visual


def load_data(dataset:datasets.Dataset)->numpy.ndarray:
  match type(dataset):
    case datasets.TwoClassMotorImagery:
      return dataset.data
    
    case datasets.FourClassMotorImagery:
      return dataset.data

    case datasets.DriveAural1:
      return dataset.data
    
    case datasets.DriveAural2:
      return dataset.data