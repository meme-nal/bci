import os
import sys
import json, ujson
import numpy
import time
from tqdm import tqdm


def make_batches(src_dir_path:str, dst_dir_path:str, batch_size:int):
  if not os.path.exists(dst_dir_path):
    os.makedirs(dst_dir_path)
  
  os.makedirs(os.path.join(dst_dir_path, "batches"))
  os.makedirs(os.path.join(os.path.join(dst_dir_path, "batches"), "meta"))

  batches_path = os.path.join(os.path.join(dst_dir_path, "batches"))
  meta_path = os.path.join(os.path.join(batches_path, "meta"))

  pi = 0 # patch index
  bi = 0 # batch index
  patch_main_data  = numpy.array([])
  patch_label_data = numpy.array([])
  for patch in tqdm(os.listdir(src_dir_path)):
    if pi >= batch_size:
      # deprecated
      #patch_data = numpy.concatenate((patch_main_data, patch_label_data), axis=0)
      #patch_data.astype('uint8').tofile(os.path.join(batches_path, 'batch_' + str(bi)))
      with open(os.path.join(batches_path, 'batch_' + str(bi)), 'wb') as f:
        print(f"patch_main_data: {patch_main_data.astype(main_dtype)} | shape: {patch_main_data.shape}")
        print(f"patch_label_data: {patch_label_data.astype(label_dtype)} | shape: {patch_label_data.shape}")
        patch_main_data.astype(main_dtype).tofile(f)
        patch_label_data.astype(label_dtype).tofile(f)

      meta = {}
      meta['main_data_shape'] = patch_json["main_data_shape"]
      meta['main_dtype'] = main_dtype
      meta['label_data_shape'] = patch_json["label_data_shape"]
      meta['label_dtype'] = label_dtype
      meta['batch_size'] = batch_size
      with open(os.path.join(meta_path, 'meta_' + str(bi)) + '.json', 'w') as fp:
        ujson.dump(meta, fp, sort_keys=False)

      pi = 0
      patch_main_data  = numpy.array([])
      patch_label_data = numpy.array([])
      bi += 1
    
    patch_json = json.load(open(os.path.join(src_dir_path, patch)))
    patch_main_data = numpy.concatenate((patch_main_data, numpy.array(patch_json["main_data"])), axis=0)
    patch_label_data = numpy.concatenate((patch_label_data, numpy.array(patch_json["label_data"])), axis=0)
    main_dtype  = patch_json["main_dtype"]
    label_dtype = patch_json["label_dtype"]
    
    pi += 1


if __name__ == '__main__':
  src_dir_path = sys.argv[1]
  dst_dir_path = sys.argv[2]
  batch_size   = int(sys.argv[3])

  start = time.time()
  make_batches(src_dir_path, dst_dir_path, batch_size)
  end   = time.time()
  print(f'\nTime: {end - start}')
