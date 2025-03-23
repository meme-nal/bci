import numpy
import mne
import matplotlib.pyplot as plt


def plot_from_file(file_path:str)->None:
    if file_path.endswith('.edf'):
        raw = mne.io.read_raw_edf(file_path, preload=True)
        raw.plot()
        plt.show()
    elif file_path.endswith('.fif'):
        raw = mne.io.read_raw_fif(file_path, preload=True)
        raw.plot()
        plt.show()
    else:
        print("This file extension is not supported")
