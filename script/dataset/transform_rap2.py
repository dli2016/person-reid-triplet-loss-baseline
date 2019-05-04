"""Refactor file directories, save/rename images and partition the 
train/val/test set, in order to support the unified dataset interface.
"""

from __future__ import print_function

import sys
sys.path.insert(0, '.')

from zipfile import ZipFile
import os.path as osp
import numpy as np
import shutil

from scipy.io import loadmat

from tri_loss.utils.utils import may_make_dir
from tri_loss.utils.utils import save_pickle
from tri_loss.utils.utils import load_pickle

from tri_loss.utils.dataset_utils import get_im_names
from tri_loss.utils.dataset_utils import partition_train_val_set
from tri_loss.utils.dataset_utils import new_im_name_tmpl
from tri_loss.utils.dataset_utils import parse_im_name as parse_new_im_name
from collections import defaultdict

def parse_original_im_name(img_name, parse_type='id'):
  """Get the person id or cam from an image name."""
  assert parse_type in ('id', 'cam')
  if parse_type == 'id':
    parsed = int(img_name[:5])
  else:
    parsed = int(img_name[9:11])
  return parsed


def _loadAnnotationFile(fname):
  """Load RAP2 in which both the ReID and Attributes are provided"""
  assert osp.exists(fname)
  # Load data
  data = loadmat(open(fname, 'r'))
  data = data['LabelData_fusion'][0][0]
  img_fname = data[0]
  identities = data[6]
  nums = identities.shape[0]
  identities = np.reshape(identities, (nums,))
  reid_partitions = data[7][0][0]
  reid_train_id = reid_partitions[0][0]
  reid_test_id = reid_partitions[1][0]
      
  return img_fname, identities, reid_train_id, reid_test_id

def _convertFNameFormat(inputs):
  """Convert the type of filename from numpy.unicode to numpy.string_"""
  num = inputs.shape[0]
  items = []
  for idx in range(num):
    item = inputs[idx][0].astype('S')
    items.append(item[0])
  items = np.asarray(items)
  return items

def _renameRAP2(fnames, ids):
  """For conveniently using the existed codes ..."""
  nfnms = fnames.shape[0]
  nids = ids.shape[0]
  assert nfnms == nids

  new_fnames = []
  for idx in range(nids):
    fname = fnames[idx]
    pid = ids[idx]
    new_name = str(pid).zfill(5) + '-' + fname
    new_fnames.append(new_name)
  new_fnames = np.asarray(new_fnames)
  return new_fnames

def _loadQueryFile(fname):
  """Load query images"""
  assert osp.exists(fname)
  data = np.loadtxt(fname, dtype=str)
  return data

def _getSelectedImgNames(fnames, total, selected):
  """Split the dataset"""
  selected_fnames = []
  corresponding_ids = []
  for pid in selected:
    pid_positions, = np.where(total==pid)
    pid_values = total[pid_positions]
    pid_fnames = fnames[pid_positions]
    selected_fnames.append(pid_fnames)
    corresponding_ids.append(pid_values)
  selected_fnames = np.hstack(selected_fnames)
  corresponding_ids = np.hstack(corresponding_ids)
  return selected_fnames, corresponding_ids

def _getGallerySet(test_set, test_id, query_set):
  """Get the gallery set from testset"""
  gallery_set = []
  gallery_id = []
  for img_test, id_test in zip(test_set, test_id):
    cond, = np.where(query_set==img_test)
    if cond.shape[0] == 0:
      gallery_set.append(img_test)
      gallery_id.append(id_test)
  return np.asarray(gallery_set), np.asarray(gallery_id)

def _get_im_names_rap2(annotation_file, query_file, flag=0):
  """Get the image names of different sets"""
  # Load necessary data
  img_fnames, identities, reid_train_id, reid_test_id =\
    _loadAnnotationFile(annotation_file)
  _convertFNameFormat(img_fnames)
  img_fnames = _convertFNameFormat(img_fnames)
  # Training images
  img_fnames_train, img_pids_train = _getSelectedImgNames(img_fnames, \
    identities, reid_train_id)
  # Test images
  img_fnames_test, img_pids_test = _getSelectedImgNames(img_fnames, \
    identities, reid_test_id)
  # QUery images
  img_fnames_query = _loadQueryFile(query_file)
  q_img_positions = []
  for q_fname in img_fnames_query:
    q_img_pos, = np.where(img_fnames==q_fname)
    q_img_positions.append(q_img_pos[0])
  q_img_positions = np.asarray(q_img_positions)
  img_pids_query = identities[q_img_positions]
  # Gallery images
  img_fnames_gallery, img_pids_gallery = _getGallerySet(img_fnames_test, \
    img_pids_test, img_fnames_query)
  if flag == 0:
    return img_fnames_train, img_fnames_test, img_fnames_query
  # Rename:
  img_fnames_train_n = _renameRAP2(img_fnames_train, img_pids_train)
  img_fnames_test_n = _renameRAP2(img_fnames_test, img_pids_test)
  img_fnames_query_n = _renameRAP2(img_fnames_query, img_pids_query)
  return img_fnames_train_n, img_fnames_test_n, img_fnames_query_n

def _move_ims(ori_im_dir, ori_img_paths_fake, new_im_dir, \
  parse_im_name, new_im_name_tmpl):
  new_im_name_frmt = '{:08d}_{:04d}_{:08d}.png'
  """Rename and move images to new directory."""
  cnt = defaultdict(int)
  new_im_names = []
  for img_path_fake in ori_img_paths_fake:
    img_path = img_path_fake[6:]
    im_name = osp.join(ori_im_dir, img_path)
    id = parse_im_name(img_path_fake, 'id')
    cam = parse_im_name(img_path_fake, 'cam')
    cnt[(id, cam)] += 1
    new_im_name = new_im_name_frmt.format(id, cam, cnt[(id, cam)] - 1)
    new_im_names.append(new_im_name)
    shutil.copy(im_name, osp.join(new_im_dir, new_im_name))
  return new_im_names

def save_images_rap2(zip_file, annotation_file, query_file, save_dir=None, \
  train_test_split_file=None):
  """Rename and move all the used images to a diretory."""
  print("Extracting zip file")
  root = osp.dirname(osp.abspath(zip_file))
  if save_dir is None:
    save_dir = root
  may_make_dir(save_dir)
  with ZipFile(zip_file) as z:
    z.extractall(path=save_dir)
  print("Extracting zip file done")

  new_im_dir = osp.join(save_dir, 'images')
  may_make_dir(new_im_dir)
  raw_dir = osp.join(save_dir, osp.basename(zip_file)[:-4])

  # Get fnames
  img_fnames_train, img_fnames_test, img_fnames_query = \
    _get_im_names_rap2(annotation_file, query_file, 1)
  img_fnames_train.sort()
  img_fnames_test.sort()
  img_fnames_query.sort()
  im_paths = list(img_fnames_train) + list(img_fnames_test) + \
    list(img_fnames_query)
  nums = [img_fnames_train.shape[0], img_fnames_test.shape[0], \
    img_fnames_query.shape[0]]

  # Move images
  org_img_dir = osp.join(root, 'images-pedestrian')
  im_names = _move_ims(org_img_dir, im_paths, new_im_dir, \
    parse_original_im_name, new_im_name_tmpl)  

  split = dict()
  keys = ['trainval_im_names', 'gallery_im_names', 'q_im_names']
  inds = [0] + nums
  inds = np.cumsum(inds)
  for i, k in enumerate(keys):
    split[k] = im_names[inds[i]:inds[i + 1]]
  save_pickle(split, train_test_split_file)
  print('Saving images done.')
  return split

def save_images(zip_file, save_dir=None, train_test_split_file=None):
  """Rename and move all used images to a directory."""

  print("Extracting zip file")
  root = osp.dirname(osp.abspath(zip_file))
  if save_dir is None:
    save_dir = root
  may_make_dir(save_dir)
  with ZipFile(zip_file) as z:
    z.extractall(path=save_dir)
  print("Extracting zip file done")

  new_im_dir = osp.join(save_dir, 'images')
  may_make_dir(new_im_dir)
  raw_dir = osp.join(save_dir, osp.basename(zip_file)[:-4])

  im_paths = []
  nums = []

  for dir_name in ['bounding_box_train', 'bounding_box_test', 'query']:
    im_paths_ = get_im_names(osp.join(raw_dir, dir_name),
                             return_path=True, return_np=False)
    im_paths_.sort()
    im_paths += list(im_paths_)
    nums.append(len(im_paths_))

  im_names = move_ims(
    im_paths, new_im_dir, parse_original_im_name, new_im_name_tmpl)

  split = dict()
  keys = ['trainval_im_names', 'gallery_im_names', 'q_im_names']
  inds = [0] + nums
  inds = np.cumsum(inds)
  for i, k in enumerate(keys):
    split[k] = im_names[inds[i]:inds[i + 1]]

  save_pickle(split, train_test_split_file)
  print('Saving images done.')
  return split


def transform(zip_file, annotation_file, query_file, save_dir=None):
  """Refactor file directories, rename images and partition the train/val/test 
  set.
  """

  train_test_split_file = osp.join(save_dir, 'train_test_split.pkl')
  # train_test_split = save_images(zip_file, save_dir, train_test_split_file)
  train_test_split = save_images_rap2(zip_file, annotation_file, query_file, \
    save_dir, train_test_split_file)
  # train_test_split = load_pickle(train_test_split_file)
  # partition train/val/test set

  trainval_ids = list(set([parse_new_im_name(n, 'id')
                           for n in train_test_split['trainval_im_names']]))
  # Sort ids, so that id-to-label mapping remains the same when running
  # the code on different machines.
  trainval_ids.sort()
  trainval_ids2labels = dict(zip(trainval_ids, range(len(trainval_ids))))
  partitions = partition_train_val_set(
    train_test_split['trainval_im_names'], parse_new_im_name, num_val_ids=100)
  train_im_names = partitions['train_im_names']
  train_ids = list(set([parse_new_im_name(n, 'id')
                        for n in partitions['train_im_names']]))
  # Sort ids, so that id-to-label mapping remains the same when running
  # the code on different machines.
  train_ids.sort()
  train_ids2labels = dict(zip(train_ids, range(len(train_ids))))

  # A mark is used to denote whether the image is from
  #   query (mark == 0), or
  #   gallery (mark == 1), or
  #   multi query (mark == 2) set

  val_marks = [0, ] * len(partitions['val_query_im_names']) \
              + [1, ] * len(partitions['val_gallery_im_names'])
  val_im_names = list(partitions['val_query_im_names']) \
                 + list(partitions['val_gallery_im_names'])

  test_im_names = list(train_test_split['q_im_names']) \
                  + list(train_test_split['gallery_im_names'])
  test_marks = [0, ] * len(train_test_split['q_im_names']) \
               + [1, ] * len(train_test_split['gallery_im_names'])

  partitions = {'trainval_im_names': train_test_split['trainval_im_names'],
                'trainval_ids2labels': trainval_ids2labels,
                'train_im_names': train_im_names,
                'train_ids2labels': train_ids2labels,
                'val_im_names': val_im_names,
                'val_marks': val_marks,
                'test_im_names': test_im_names,
                'test_marks': test_marks}
  partition_file = osp.join(save_dir, 'partitions.pkl')
  save_pickle(partitions, partition_file)
  print('Partition file saved to {}'.format(partition_file))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(
    description="Transform RAP2 Dataset")
  parser.add_argument('--zip_file', type=str,
                      default='~/Dataset/duke/RAP2.zip')
  parser.add_argument('--save_dir', type=str,
                      default='~/Dataset/duke')
  args = parser.parse_args()
  #zip_file = osp.abspath(osp.expanduser(args.zip_file))
  #save_dir = osp.abspath(osp.expanduser(args.save_dir))
  #transform(zip_file, save_dir)
  # Test ...
  annotation_file = './data/LabelData_fusion_v1_v2.mat'
  query_file = './data/query_test_image_name.txt'
  zip_file = './data/images-pedestrian.zip'
  save_dir = './data'
  transform(zip_file, annotation_file, query_file, save_dir)
