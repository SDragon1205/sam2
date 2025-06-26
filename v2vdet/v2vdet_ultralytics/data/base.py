# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional
import lmdb
import pickle

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from ultralytics.data.base import BaseDataset

class BaseDatasetV2(BaseDataset):
    """
    Base dataset can read image from lmdb database.
    """
    
    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__(img_path, 
                         imgsz, 
                         cache, 
                         augment, 
                         hyp, 
                         prefix, 
                         rect, 
                         batch_size, 
                         stride, 
                         pad, 
                         single_cls, 
                         classes, 
                         fraction)

        self.lmdb_dirs = self.get_lmdb()
    
    def get_lmdb(self):
        lmdb_dirs = set()
        flag = False
        for path in self.im_files:
            dir_path = os.path.dirname(path)           
            if os.path.isdir(f"{dir_path}.lmdb"):
                lmdb_dirs.add(f"{dir_path}.lmdb")
                flag = True
            
        if flag is False: 
            return None
        else:     
            return list(lmdb_dirs)
    
    def read_from_lmdb(self, lmdb_path, rel_path):
        """
        Read the image from lmdb database.
        
        Args:
        - lmdb_path: LMDB dataset path
        - rel_path: The relative path of the image to be read (Only the base name of the img.)
        
        Returns:
        - The image as a numpy array.
        """
        env = lmdb.open(lmdb_path, 
                        readonly=True,
                        lock=False, 
                        readahead=False,
                        meminit=False
                        )
        with env.begin() as txn:
            key = rel_path.encode('utf-8')
            value = txn.get(key)
            
            if value is None:
                env.close()
                return None
            
            img = pickle.loads(value)
        
        env.close()
        # LOGGER.info(f"Successfully read image from lmdb: {lmdb_path} {rel_path}!")
        return img

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)  # BGR
            elif os.path.exists(f):
                im = cv2.imread(f)  # BGR
            elif hasattr(self, 'lmdb_dirs'):
                if self.lmdb_dirs is not None:  # read image from lmdb
                    image_name = os.path.basename(f)
                    for lmdb_database in self.lmdb_dirs:
                        im = self.read_from_lmdb(lmdb_database, image_name)
                        if im is not None:
                            break
                    if im is None:
                        LOGGER.warning(f"{f} not found in lmdb database, read image from disk")
                        im = cv2.imread(f)
            else:  # read image
                im = cv2.imread(f)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]