import numpy as np
from torch.utils.data import Dataset

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomImageDataset(Dataset):
    def __init__(self, samples, distances):
        self.samples = np.loadtxt(samples)
        self.distances = np.loadtxt(distances)
        self.samples = self.samples.reshape((len(self.distances), 3))
        #Oder
        #self.samples = self.samples.reshape((len(self.samples)/3, 3))

    def __len__(self):
        return len(self.distances)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        distance = self.distances[idx]
        return sample, distance
