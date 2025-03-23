import sys
import numpy as np
import mne
import matplotlib.pyplot as plt


def main(path_to_edf:str)->None:
    raw = mne.io.read_raw_edf(path_to_edf, preload=True)
    annotations = raw.annotations
    print(annotations)


if __name__ == '__main__':
    path_to_edf:str = sys.argv[1]
    main(path_to_edf)
