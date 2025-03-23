import sys
import mne
import matplotlib.pyplot as plt


def main(src_edf_path:str, dst_edf_path:str)->None:
    raw = mne.io.read_raw_edf(src_edf_path, preload=True)

    raw.plot()
    plt.show()

    annotations = raw.annotations
    raw.set_annotations(annotations)
    raw.save(dst_edf_path, overwrite=True)


if __name__ == '__main__':
    src_edf_path:str = sys.argv[1]
    dst_edf_path:str = sys.argv[2]
    main(src_edf_path, dst_edf_path)
