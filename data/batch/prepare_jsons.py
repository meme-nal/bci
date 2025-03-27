import os
import sys
from PIL import Image
import time
import ujson
import numpy
from tqdm import tqdm


round_val = lambda x: round(x, 6)
round_func = numpy.vectorize(round_val)


# TODO: make parallel
def make_time_from_text(src_dir_path:str, dst_dir_path:str, formats:str)->None:
  window_size = 5
  pi = 0
  if not os.path.exists(dst_dir_path):
    os.makedirs(dst_dir_path)

  main_format, label_format = formats.split(',') # "float32,float32"

  for filename in tqdm(os.listdir(src_dir_path)):
    datap = os.path.join(src_dir_path, filename)
    data_values = numpy.loadtxt(datap, delimiter=',')
    data_patches = []
    for i in tqdm(range(data_values.shape[0] - window_size)):
      channel_num = data_values.shape[1]

      data = {}
      data['main_data'] = list(round_func(data_values[i:i+window_size,:]).ravel())
      data['main_data_shape'] = [1, window_size, channel_num]
      data['main_dtype'] = main_format
      data['label_data'] = list(round_func(data_values[i+1:i+1+window_size,:]).ravel())
      data['label_data_shape'] = [1, window_size, channel_num]
      data['label_dtype'] = label_format

      dst_file_path = os.path.join(dst_dir_path, os.path.basename(datap).split('.')[0] + '_' + str(pi) + '.json')
      with open(dst_file_path, 'w') as fp:
        ujson.dump(data, fp, sort_keys=False)
      pi += 1


# TODO: make parallel
def make_from_text(src_dir_path:str, dst_dir_path:str, formats:str):
  patch_size = 100
  if not os.path.exists(dst_dir_path):
    os.makedirs(dst_dir_path)

  main_format, label_format = formats.split(',') # "float32,float32"

  for filename in tqdm(os.listdir(src_dir_path)):
    datap = os.path.join(src_dir_path, filename)
    data_values = numpy.loadtxt(datap, delimiter=',')
    data_patches = []
    for i in range(data_values.shape[0]):
      if (i+1) % patch_size == 0:
        data_patches.append(data_values[i-patch_size+1:i+1])

    pi = 0
    for patch in data_patches:
      data = {}
      data['main_data'] = list(patch[:,:-1].ravel())
      data['main_data_shape'] = [patch_size, patch[:,:-1].shape[1]]
      data['main_dtype'] = main_format
      data['label_data'] = list(patch[:,-1])
      data['label_data_shape'] = [patch_size, patch[:,[-1]].shape[1]]
      data['label_dtype'] = label_format

      dst_file_path = os.path.join(dst_dir_path, os.path.basename(datap).split('.')[0] + '_' + str(pi) + '.json')
      with open(dst_file_path, 'w') as fp:
        ujson.dump(data, fp, sort_keys=False)
      pi += 1



# TODO: make parallel 
def make_from_img(src_dir_path:str, dst_dir_path:str, formats:str):
  if not os.path.exists(dst_dir_path):
    os.makedirs(dst_dir_path)

  main_format, label_format = formats.split(',') # "uint8,float32"

  for filename in tqdm(os.listdir(src_dir_path)):
    imgp = os.path.join(src_dir_path, filename)
    img = Image.open(imgp, 'r')
    pixel_values = list(img.getdata())
    pixel_values_flat = [x for sets in pixel_values for x in sets]

    data = {}
    data['main_data'] = pixel_values_flat
    data['main_data_shape'] = [3, *img.size]
    data['main_dtype'] = main_format
    data['label_data'] = [1]
    data['label_data_shape'] = [1, 1, 1]
    data['label_dtype'] = label_format

    dst_file_path = os.path.join(dst_dir_path, os.path.basename(imgp).split('.')[0] + '.json')
    with open(dst_file_path, 'w') as fp:
      ujson.dump(data, fp, sort_keys=False)


if __name__ == '__main__':
  src_dir_path = sys.argv[1]
  dst_dir_path = sys.argv[2]
  formats      = sys.argv[3]

  start = time.time()
  make_time_from_text(src_dir_path, dst_dir_path, formats)
  end   = time.time()
  print(f'\nTime: {end - start}')
