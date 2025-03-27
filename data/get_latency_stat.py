import os, sys
import numpy as np
import mne


###
# PARAMS
##

files_dont_open:list[str] = ['hand_motor_right_2.fif']


def main(src_dir_path:str)->None:
  latencies:dict[str, list[float]] = {}
  avg_latencies:dict[str, float] = {}

  for filename in os.listdir(src_dir_path):
    if filename in files_dont_open:
      continue

    file_path = os.path.join(src_dir_path, filename)
    raw = mne.io.read_raw_fif(file_path, preload=True)

    annotations = raw.annotations

    for annotation in annotations:
      onset = annotation['onset']
      duration = annotation['duration']
  
      start_sample = raw.time_as_index(onset)[0]
      end_sample = raw.time_as_index(onset + duration)[0]
  
      eeg_epoch_data, times = raw[:, start_sample:end_sample]
  
      if not str(annotation["description"]) in latencies:
          latencies[str(annotation["description"])] = []
      latencies[str(annotation["description"])].append(eeg_epoch_data.shape[1])

  for key in latencies.keys():
    avg_latencies[key] = np.mean(latencies[key])

  print("\n==============================\n")
  for key in avg_latencies.keys():
    print(f"{key} : {avg_latencies[key]}")


if __name__ == '__main__':
  src_dir_path:str = sys.argv[1]

  main(src_dir_path)

