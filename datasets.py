import numpy
import os, sys
import urllib


class Dataset:
  def __init__(self):
    pass


class FourClassMotorImagery(Dataset):
  def __init__(self):
    self.data = numpy.array([])

    '''
    download data from some url into tmp dir
    process data, convert into numpy array and load into self.data
    '''


class TwoClassMotorImagery(Dataset):
  def __init__(self):
    self.data = numpy.array([])


class DriveAural1(Dataset):
  def __init__(self):
    self.data = numpy.array([])


class DriveAural2(Dataset):
  def __init__(self):
    self.data = numpy.array([])
