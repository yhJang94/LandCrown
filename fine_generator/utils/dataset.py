import os
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm.auto import tqdm
import open3d as o3d
import pickle
import pdb

    
    
class CrownDataset(Dataset):
    def __init__(self, input_path, cr_path, subset='train'):
        
        self.input_path = input_path
        self.cr_path = cr_path
        self.subset = subset
        
        self.base = os.path.dirname(input_path)
        self.data_list_file = os.path.join(self.base, f'{self.subset}.txt')

        with open(self.data_list_file, 'r') as f:
            self.file_list = [line.strip() for line in f if line.strip()]
            
        self.mapping = {
                0: 0,
                11: 7, 12: 6, 13: 5, 14: 4, 15: 3, 16: 2, 17: 1,
                21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                31: 7, 32: 6, 33: 5, 34: 4, 35: 3, 36: 2, 37: 1,
                41: 8, 42: 9, 43: 10, 44: 11, 45: 12, 46: 13, 47: 14
            }
        
        self.lut = np.zeros(48, dtype=np.int64)
        for fdi, new_idx in self.mapping.items():
            self.lut[fdi] = new_idx
        
    def remap_labels(self, label_array):
        return self.lut[label_array.astype(np.int64)]
        
    def normalize(self, input, target, fix=True):
        
        centroid = np.mean(input, axis=0, keepdims=True)
        input = input - centroid
        new_target = target - centroid
        max_abs = np.max(np.sqrt(np.sum(input ** 2, axis=1)))
        input = input / max_abs
        new_target = new_target / max_abs

        if fix:
            return input, new_target
        else:
            return input, new_target.reshape(12, -1, 3)
        

        
    def augmentation(self, input, target, fix=True):
        
        theta = np.random.uniform(-np.pi/36, np.pi/36)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0,      0,     1]
        ])


        rotated_input = input @ R.T
        rotated_target = target @ R.T
        

        
        scale = np.random.uniform(0.9, 1.0)
        noise = np.random.normal(0, 0.002, size=rotated_input.shape)
        
        rotated_input = rotated_input * scale + noise
        rotated_target = rotated_target * scale

        if fix:
            return rotated_input, rotated_target
        else:
            return rotated_input, rotated_target.reshape(12, -1, 3)
    
    def tooth_to_index(self, tooth_num):
        quadrant = tooth_num // 10     
        position = tooth_num % 10       
        return (quadrant - 1) * 7 + (position - 1)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        name = sample.split('.')[0]

        input_data = IO.get(os.path.join(self.input_path, name + '.npy'))
        down_data = IO.get(os.path.join(self.cr_path, name + '.npy'))
        # already norm
        bbox = IO.get(os.path.join(self.cr_path.replace("data_re_448", "bbox"), name + '.npy'))
        
        input_data, down_data = self.normalize(input_data, down_data, fix=True)
        
        if self.subset == 'train':
            input_data, down_data = self.augmentation(input_data, down_data, fix=True)     
        
        idx = int(name.split('_')[2])

        start = idx * 1024
        end = (idx + 1) * 1024

        partial = np.concatenate([input_data[:start], input_data[end:]], axis=0)
        gt = input_data[start:end]
        
        start = idx * 32
        end = (idx + 1) * 32
        
        coarse_partial = np.concatenate([down_data[:start], down_data[end:]], axis=0)
        coarse_pred = down_data[start:end]
    
        partial = torch.from_numpy(partial).float()
        gt = torch.from_numpy(gt).float()
        coarse_partial = torch.from_numpy(coarse_partial).float()
        coarse_pred = torch.from_numpy(coarse_pred).float()
        idx = torch.tensor(idx, dtype=torch.long)
        bbox = torch.from_numpy(bbox).float()
        
        full_range = torch.arange(28)
        mask = ~(full_range[:, None] == idx).any(dim=1)
        cond_idx = full_range[mask]

        
        return {'partial': partial, 'gt': gt, 'missing_idx': idx, 'cond_idx': cond_idx, 'coarse_partial': coarse_partial, 'coarse_pred': coarse_pred, 'ori': input_data,
                'bbox': bbox}
    def __len__(self):
        return len(self.file_list)


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.npz']:
            return cls._read_npz(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]
    
    @classmethod
    def _read_npz(cls, file_path):
        f = np.load(file_path, allow_pickle=True)
        try:
            return f["data"].item()
        except:
            return f