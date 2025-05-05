import sys
import os
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.o3d_tools import *


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.files = []

    def __len__(self):
        return len(self.files)


class CollationFunctionFactory:
    def __init__(self):
        pass

    def __call__(self, list_data):
        return self.collate_default(list_data)

    def collate_default(self, list_data):
        if len(list_data) > 1:
            return self.collate_tuple(list_data)
        else:
            return list_data

    def collate_tuple(self, list_data):
        outputs = []
        for batch_data in list_data:
            contrastive_tuple = []
            for tuple_data in batch_data:
                if isinstance(tuple_data, np.ndarray):
                    contrastive_tuple.append(tuple_data)
                elif isinstance(tuple_data, list):
                    contrastive_tuple.extend(tuple_data)
        outputs = [torch.from_numpy(ct).float() for ct in contrastive_tuple]
        outputs = torch.stack(outputs)
        return outputs
