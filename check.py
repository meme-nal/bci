import os, sys
import json, ujson
import time
import numpy as np
import bci
from bci import load_data

round_val = lambda x: round(x, 6)
round_func = np.vectorize(round_val)
#np.set_printoptions(precision = 10, suppress = True)
#format(num, '.8f')

def main(src_dir_path:str, dst_dir_path:str, formats:str)->None:
  #datap = '/home/me_me/eeg_marked/hand_motor_right.fif'
  #bci.processing.remove_bad_channels(src_dir_path)
  #bci.processing.rename_channels(src_dir_path, ['EEG CZ-A2_CZ-A2'], ['EMG'])
  #bci.visual.plot_from_file(datap)
  raw_epochs_data = bci.processing.split_on_epochs(src_dir_path, [90, 90])
  #print(raw_epochs_data['motor'].shape)
  #print(raw_epochs_data['non_stimulus'].shape)

  main_format, label_format = formats.split(',') # "float32,float32"

  c0 = raw_epochs_data['non_stimulus']
  c0_label = np.array([[0]]*c0.shape[0])

  c1 = raw_epochs_data['motor']
  c1_label = np.array([[1]]*c1.shape[0], dtype='U')

  all_data = np.concatenate((c0, c1), axis=0)
  all_data_normalized = bci.processing.normalize(all_data, 2) # [num_epochs, num_channels, num_time_records]

  all_labels = np.concatenate((c0_label, c1_label), axis=0)
  
  indices = np.random.permutation(all_data_normalized.shape[0])

  shuffled_all_data = all_data_normalized[indices]
  shuffled_all_labels = all_labels[indices]
  
  for pi in range(shuffled_all_data.shape[0]):
    data = {}
    data['main_data'] = list(round_func(shuffled_all_data[pi].ravel()))
    data['main_data_shape'] = [1, shuffled_all_data.shape[1], shuffled_all_data.shape[2]]
    data['main_dtype'] = main_format
    data['label_data'] = list(shuffled_all_labels[pi])
    #data['label_data_shape'] = [1, 1, shuffled_all_labels.shape[1]]
    data['label_data_shape'] = [1, 1, 1]
    data['label_dtype'] = label_format

    dst_file_path = os.path.join(dst_dir_path, 'patch' + '_' + str(pi) + '.json')
    with open(dst_file_path, 'w') as fp:
      ujson.dump(data, fp, sort_keys=False)


if __name__ == '__main__':
  #raw_data = load_data(bci.datasets.MotorEeg('/home/me_me/tmp'))
  #print(raw_data)
  src_dir_path = sys.argv[1]
  dst_dir_path = sys.argv[2]
  formats      = sys.argv[3]

  start = time.time()
  main(src_dir_path, dst_dir_path, formats)
  end   = time.time()
  print(f'\nTime: {end - start}')
