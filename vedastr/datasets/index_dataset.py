import os

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class IndexDataset(BaseDataset):

    def __init__(self, root, gt_txt, transform, character,
                 batch_max_length, *args, **kwargs):
        super(IndexDataset, self).__init__(root, gt_txt, transform, character=character,
                                         batch_max_length=batch_max_length,
                                          *args, **kwargs)

    def get_name_list(self):
        with open(self.gt_txt, 'r') as gt:
            for line in gt.readlines():
                tmp = line.strip('\n').split(' ', 1)
                img_name, label = tmp[0], tmp[-1]
                if self.filter(label):
                    continue
                else:
                    self.img_names.append(os.path.join(self.root, img_name))
                    self.gt_texts.append(label)

        self.samples = len(self.gt_texts)
