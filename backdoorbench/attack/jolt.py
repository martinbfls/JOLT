'''
JOLT / FLIP label-flip backdoor for BackdoorBench, replayed from an existing JOLT run.

Nothing is optimized here: the flipped labels come from a JOLT select_flips output (single_user
branch, i.e. the centralized deployment) and the trigger from the joint labels+trigger
optimization. BackdoorBench then trains its own victim on that poisoned training set and saves
the usual record/<name>/attack_result.pt, so every BackdoorBench detector/defense runs unchanged.

  - bd_train: the full clean training set where the flipped samples keep their CLEAN image and
    get the JOLT label (poison_indicator = flipped indices) -- exactly what JOLT's user trains on.
  - bd_test: source-class test images + the trigger (x + delta, clipped to [0, 1], as JOLT's
    OptimizedPoisoner), labeled with the target class.

Lives in the JOLT repo (backdoorbench/attack/jolt.py) and is copied into a BackdoorBench
checkout by backdoorbench/run_backdoorbench.sh; it imports BackdoorBench's own modules.
'''

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path = ["./"] + sys.path

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchvision.transforms import ToPILImage, ToTensor  # noqa: E402

from attack.badnet import BadNet, add_common_attack_args  # noqa: E402
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform  # noqa: E402


class AddTrigger:
    """x -> clip(x + delta, 0, 1) in [0, 1] pixel space, delta resized to the image if needed."""

    def __init__(self, trigger_path):
        self.delta = torch.load(trigger_path, map_location="cpu").detach().float()

    def __call__(self, img, target=None, image_serial_id=None):
        x = ToTensor()(img)
        delta = self.delta
        if delta.shape[-2:] != x.shape[-2:]:
            delta = F.interpolate(delta.unsqueeze(0), size=x.shape[-2:], mode="bilinear", align_corners=False)[0]
        return ToPILImage()((x + delta).clamp(0, 1))


class JOLT(BadNet):

    def set_bd_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        parser.add_argument("--flips_dir", type=str,
                            help="JOLT select_flips output directory of a single_user (1-poisoned/0-honest) branch")
        parser.add_argument("--budget", type=int, help="flip budget to replay (one of select_flips' budgets)")
        parser.add_argument("--trigger_path", type=str, help="JOLT trigger .pt (delta in [0, 1] pixel space)")
        parser.add_argument("--source_label", type=int, help="class whose test images are triggered for the ASR")
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/jolt/default.yaml',
                            help='path for yaml file provide additional default attributes')
        return parser

    def stage1_non_training_data_prepare(self):
        logging.info("stage1 start (JOLT replay, no attack optimization)")
        args = self.args
        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform, \
        clean_train_dataset_with_transform, \
        clean_train_dataset_targets, \
        clean_test_dataset_with_transform, \
        clean_test_dataset_targets \
            = self.benign_prepare()

        flips_dir = Path(args.flips_dir)
        clean_targets = np.asarray(clean_train_dataset_targets)
        true = np.load(flips_dir / "true.npy").argmax(axis=1)
        if not np.array_equal(true, clean_targets):
            raise ValueError(f"{flips_dir}/true.npy does not match BackdoorBench's {args.dataset} training labels "
                             "-- the JOLT run was made on another dataset or sample order.")

        idx = np.load(flips_dir / f"worker0/{args.budget}_indices.npy")
        labels = np.load(flips_dir / f"worker0/{args.budget}_labels.npy").argmax(axis=1)  # = train_user's hard labels
        if len(np.unique(idx)) != len(clean_targets):
            raise ValueError(f"{flips_dir}/worker0 holds {len(np.unique(idx))} of {len(clean_targets)} samples: "
                             "use a single_user (1-poisoned/0-honest) select_flips branch.")
        jolt_labels = clean_targets.copy()
        jolt_labels[idx] = labels
        flipped = np.where(jolt_labels != clean_targets)[0]
        logging.info(f"JOLT budget {args.budget}: {len(flipped)} flipped labels "
                     f"(target-class share {np.mean(jolt_labels[flipped] == args.attack_target) if len(flipped) else 0:.2f})")

        bd_train_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(train_dataset_without_transform),
            save_folder_path=f"{args.save_path}/bd_train_dataset",
        )
        for i in flipped:  # clean image, JOLT label
            img, label = train_dataset_without_transform[i]
            bd_train_dataset.set_one_bd_sample(int(i), img, int(jolt_labels[i]), int(label))
        torch.save(bd_train_dataset.poison_indicator, args.save_path + '/train_poison_index_list.pickle')
        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            bd_train_dataset, train_img_transform, train_label_transform,
        )

        test_poison_index = (np.asarray(clean_test_dataset_targets) == args.source_label).astype(int)
        bd_test_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(test_dataset_without_transform),
            poison_indicator=test_poison_index,
            bd_image_pre_transform=AddTrigger(args.trigger_path),
            bd_label_pre_transform=lambda label: args.attack_target,
            save_folder_path=f"{args.save_path}/bd_test_dataset",
        )
        bd_test_dataset.subset(np.where(test_poison_index == 1)[0])
        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset, test_img_transform, test_label_transform,
        )

        self.stage1_results = clean_train_dataset_with_transform, \
                              clean_test_dataset_with_transform, \
                              bd_train_dataset_with_transform, \
                              bd_test_dataset_with_transform


if __name__ == '__main__':
    attack = JOLT()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()
