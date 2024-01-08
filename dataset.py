import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import cv2
from process_data import create_house_from_name
from tqdm import tqdm


class PointCloudDataset(Dataset):
    def __init__(self, root_path):
        self.data_list = glob.glob(os.path.join(root_path, '*.npy'))
        self.cache = {}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if idx in self.cache:
            points = self.cache[idx]
        else:
            data_path = self.data_list[idx]
            data = np.load(data_path, allow_pickle=True).tolist()
            x = data['x'] # 16x64x64
            meta = data['meta']
            room_num = meta['room_num']
            dist = np.abs(x[:room_num]).min(0) # unsigned
            mask = dist < 1

            points = np.where(mask)
            points = np.stack(points) # 2xN
            points = points / 64. * 2 - 1 # [-1,1]
            points = np.concatenate((points, np.ones((1, points.shape[1]))), axis=0) # 3xN
            self.cache[idx] = points

        if points.shape[1] > 512:
            # shuffle and sample
            indices = np.arange(points.shape[1])
            np.random.shuffle(indices)
            indices = indices[:512]
            points = points[:, indices]
        else:
            pad_points = np.ones((3, 512-points.shape[1])) * -1
            points = np.concatenate([points, pad_points], axis=1)
        
        points = torch.from_numpy(points).float()

        return {'x':points}


class PointSetDataset(Dataset):
    def __init__(self, root_path, config):
        data_path_list = glob.glob(os.path.join(root_path, '*.json'))[:10]
        self.config = config
        self.data_list = []
        for data_path in tqdm(data_path_list):
            self.data_list.append(create_house_from_name(data_path, 64))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        house = self.data_list[idx]
        print(house)

if __name__ == '__main__':
    dataset = PointCloudDataset('data_bank')
    max_n = 0
    for i in range(len(dataset)):
        n = dataset[i].shape[-1]
        if n > max_n:
            max_n = n
    print(max_n)
    #dataset = PointSetDataset('data_bank', None)
    #dataset[0]