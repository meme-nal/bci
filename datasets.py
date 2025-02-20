import numpy
import os, sys
import urllib
import requests
import gdown
import mne


class Dataset:
  def __init__(self):
    pass


class FourClassMotorImagery(Dataset):
  def __init__(self, output_dir:str)->None:
    self.data = numpy.array([])

    '''
    download data from some url into tmp dir
    process data, convert into numpy array and load into self.data
    '''


class TwoClassMotorImagery(Dataset):
  def __init__(self, output_dir:str)->None:
    self.data = numpy.array([])


class DriveAural1(Dataset):
  def __init__(self, output_dir:str)->None:
    url:str = 'https://drive.google.com/uc?id=1Jtx6_9s_WtJ4iUJj-d0v_BjJLNhktTC5'
    output_filename:str = 'aural_eeg.edf'

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    if not os.path.isfile(os.path.join(output_dir, output_filename)):
      gdown.download(url, os.path.join(output_dir, output_filename), quiet=False)

    raw_data:mne.io.Raw = mne.io.read_raw_edf(os.path.join(output_dir, output_filename))
    self.data, _ = raw_data[:]


class DriveAural2(Dataset):
  def __init__(self, output_dir:str)->None:
    url:str = 'https://drive.google.com/uc?id=1bHjmpVouQXIxdsGYvNNv73JvDugYiaXi'
    output_filename:str = 'aural2_eeg.edf'

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    if not os.path.isfile(os.path.join(output_dir, output_filename)):
      gdown.download(url, os.path.join(output_dir, output_filename), quiet=False)

    raw_data:mne.io.Raw = mne.io.read_raw_edf(os.path.join(output_dir, output_filename))
    self.data, _ = raw_data[:]
