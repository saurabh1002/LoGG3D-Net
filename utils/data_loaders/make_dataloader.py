import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from torch.utils.data.sampler import Sampler
from utils.data_loaders.mulran.mulran_dataset import *


class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
        shuffle: use random permutation
    """

    def __init__(self, data_source):
        self.data_source = data_source
        self.reset_permutation()

    def reset_permutation(self):
        perm = torch.arange(len(self.data_source))
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop(0)

    def __len__(self):
        return len(self.data_source)


def make_data_loader(config):
    dset = MulRanDataset(config=config)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        collate_fn=CollationFunctionFactory(),
        num_workers=1,
        sampler=RandomSampler(dset),
        pin_memory=True,
    )
    return loader, dset.gt_poses, dset.scan_timestamps


if __name__ == "__main__":
    from config.eval_config import get_config_eval
    from utils.o3d_tools import *

    logger = logging.getLogger()
    cfg = get_config_eval()

    test_loader = make_data_loader(cfg)

    iterator = test_loader.__iter__()
    # s = iterator.size()
    l = len(test_loader.dataset)
    for i in range(l):
        input_data = next(iterator)
        quad = True
        if quad:
            visualize_scan_open3d(input_data[0][1][0][:, :3])
            visualize_scan_open3d(input_data[0][2][0][:, :3])
            visualize_scan_open3d(input_data[0][3][:, :3])
        else:
            visualize_scan_open3d(input_data[0][0][:, :3])
    print("")
