import os, sys
import numpy as np
import scipy
import mne


def rename_channels(src_dir_path:str, old_ch_names:list[str], new_ch_names:list[str])->None:
    channel_renaming = dict(zip(old_ch_names, new_ch_names))
    for filename in os.listdir(src_dir_path):
        file_path = os.path.join(src_dir_path, filename)
        raw = mne.io.read_raw_fif(file_path, preload=True)
        raw.rename_channels(channel_renaming)

        raw.save(file_path, overwrite=True)


def remove_bad_channels(src_dir_path:str)->None:
    for filename in os.listdir(src_dir_path):
        file_path = os.path.join(src_dir_path, filename)
        raw = mne.io.read_raw_fif(file_path, preload=True)

        raw.drop_channels(raw.info['bads'])

        raw.save(file_path, overwrite=True)


def split_on_epochs(src_dir_path:str, window_size:list[int])->dict[str, np.ndarray]:
    data:dict[str, list[np.ndarray]] = {}
    for filename in os.listdir(src_dir_path):
        file_path = os.path.join(src_dir_path, filename)
        raw = mne.io.read_raw_fif(file_path, preload=True)

        channel_names = raw.info["ch_names"]
        channel_names.remove('EMG')
        annotations = raw.annotations

        for annotation in annotations:
            onset = annotation['onset']

            center_sample = raw.time_as_index(onset)[0]
            start_sample = center_sample - window_size[0]
            end_sample = center_sample + window_size[1]

            eeg_epoch_data, times = raw[channel_names, start_sample:end_sample]

            if not str(annotation["description"]) in data:
                data[str(annotation["description"])] = []
            data[str(annotation["description"])].append(eeg_epoch_data)

    for key in data.keys():
        data[key] = np.array(data[key])
    return data

