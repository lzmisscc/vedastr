from eval import Ev
from string_distance.edit_distance import levenshtein
import time
import tqdm
from vedastr.utils import Config
from vedastr.runners import InferenceRunner
import cv2
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))


class ocr:
    def __init__(self, config, weights) -> None:
        super().__init__()
        self.config = config
        self.weights = weights
        cfg = Config.fromfile(self.config)

        deploy_cfg = cfg['deploy']
        common_cfg = cfg.get('common')
        cfg['batch_max_length'] = 40
        runner = InferenceRunner(deploy_cfg, common_cfg)
        runner.load_checkpoint(self.weights)
        self.runner = runner

    def run(self, im):
        pred_str, probs = self.runner(im)
        return pred_str


if __name__ == '__main__':
    ev = Ev()
    run = ocr("configs/small_satrn.py",
              "workdir/small_satrn/best_acc.pth", ).run
    table_ocr_txt_path = "../table_ocr/abs_val.txt"
    with open(table_ocr_txt_path, "r") as f:
        gt_lines = f.readlines()
    for index, line in enumerate(gt_lines):
        name, value = line.strip("\n").split("\t")
        im = cv2.imread(name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        start = time.time()
        pre = ''.join(run(im))
        print(f"{time.time()-start:.2f}\t{pre}\t{value}")
        # ev.count(value, ''.join(pre))
        # print(f"{time.time()-start:.2f}\t{ev.socre()}")
