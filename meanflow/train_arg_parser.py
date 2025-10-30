# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import logging


logger = logging.getLogger(__name__)


def get_args_parser():
    # Parse tuple for dim_scales and input_shape
    def parse_int_tuple(s):
        try:
            # Remove brackets, spaces, convert to integers
            return tuple(map(int, s.strip().strip('()').replace(' ', '').split(',')))
        except ValueError:
            raise argparse.ArgumentTypeError("Tuple must be a string of integers separated by commas, like '1, 2, 3'.")

    # Parse tuple for split_ratio    
    def parse_float_tuple(s):
        try:
            return tuple(map(float, s.strip().strip('()').replace(' ', '').split(',')))
        except ValueError:
            raise argparse.ArgumentTypeError("Tuple must be a string of numbers separated by commas, like '0.7, 0.1, 0.2'.")

    parser = argparse.ArgumentParser("Image dataset training", add_help=False)

    # Optimizer parameters
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU (effective batch size is batch_size * # gpus")
    parser.add_argument("--epochs", default=4000, type=int)
    parser.add_argument("--lr", default=0.0006, type=float, help="learning rate (absolute lr)")
    parser.add_argument("--optimizer_betas", default=[0.9, 0.999], nargs="+", type=float, help="beta1 and beta2 for Adam optimizer")
    parser.add_argument("--warmup_epochs", default=200, type=int, help="Number of warmup epochs.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate.")

    parser.add_argument("--ema_decay", default=0.9999, type=float, help="Exponential moving average decay rate.")
    parser.add_argument("--ema_decays", default=[0.99995, 0.9996], nargs="+", type=float, help="Extra EMA decay rates.")

    # Dataset parameters
    parser.add_argument("--dataset", default='cifar10', type=str, choices=['cifar10', 'mnist', 'shrimp'], help="Dataset to use.")
    parser.add_argument("--data_path", default="./data", type=str, help="data root folder with train, val and test subfolders")

    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--fid_samples", default=50000, type=int, help="number of synthetic samples for FID evaluations")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch (used when resumed from checkpoint)")
    parser.add_argument("--eval_only", action="store_true", help="No training, only run evaluation")
    parser.add_argument("--eval_frequency", default=50, type=int, help="Frequency (in number of epochs) for running FID evaluation. -1 to never run evaluation.")
    parser.add_argument("--compute_fid", action="store_true", help="Whether to compute FID in the evaluation loop. When disabled, the evaluation loop still runs and saves snapshots, but skips the FID computation.")
    parser.add_argument("--save_fid_samples", action="store_true", help="Save all samples generated for FID computation.")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--log_per_step", default=100, type=int, metavar="N", help="Log training stats every N iterations",)

    # Distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    # MeanFlow specific parameters
    parser.add_argument("--ratio", default=0.75, type=float, help="Probability of sampling r (or h) DIFFERENT from t")  

    parser.add_argument("--tr_sampler", default="v1", type=str, choices=["v0", "v1"], help="Joint (t, r) sampler version.")

    parser.add_argument("--P_mean_t", default=-0.6, type=float, help="P_mean_t of lognormal sampler.")
    parser.add_argument("--P_std_t", default=1.6, type=float, help="P_std_t of lognormal sampler.")
    parser.add_argument("--P_mean_r", default=-4.0, type=float, help="P_mean_r of lognormal sampler.")
    parser.add_argument("--P_std_r", default=1.6, type=float, help="P_std_r of lognormal sampler.")
    
    parser.add_argument("--norm_p", default=0.75, type=float, help="Norm power for adaptive weight.")
    parser.add_argument("--norm_eps", default=1e-3, type=float, help="Small constant for adaptive weight division.")
    parser.add_argument("--arch", default="unet", type=str, choices=["unet",], help="Architecture to use.")
    parser.add_argument("--use_edm_aug", action="store_true", dest="use_edm_aug", default=False, help="Enable EDM augmentation with augment labels as conditions.")

    # Debugging settings
    parser.add_argument("--test_run", action="store_true", help="Only run one batch of training and evaluation.")
    parser.add_argument("--not_compile", action="store_false", dest="compile", default=True, help="Disable compilation.")

    # shrimp
    parser.add_argument("--sat-files-path", default="", type=str, help="Path to satellite image data directory.")
    parser.add_argument("--radar-files-path", default="", type=str, help="Path to radar reflectivity image data directory.")
    parser.add_argument("--start-date", default="", type=str, help="Start date for dataset selection (e.g., 20210101).")
    parser.add_argument("--end-date", default="", type=str, help="End date for dataset selection (e.g., 20210430).")
    parser.add_argument("--max-folders", default=None, type=int, help="Maximum number of folders (days) to load. Use None to load all.")
    parser.add_argument("--history-frames", default=0, type=int, help="Number of past frames to use as input (set as 0 to use the current frame only).")
    parser.add_argument("--future-frame", default=0, type=int, help="Predict one future frame.")
    parser.add_argument("--refresh-rate", default=10, type=int, help="Time interval (in minutes) between frames.")
    parser.add_argument("--coverage-threshold", default=0.05, type=float, help="Minimum radar reflectivity coverage threshold for selecting a valid frame (0.0 to 1.0).")
    #parser.add_argument("--seed", default=96, type=int, help="Random seed for dataset buiding.")
    parser.add_argument("--block-size", default=100, type=int, help="Number of sat-radar pairs to include per data segment.")
    parser.add_argument("--split-ratio", default=(0.7, 0.1, 0.2), type=parse_float_tuple, help="Train/val/test split ratio (three floats in [0,1] that sum <= 1.0), e.g. 0.7, 0.1, 0.2.")
    parser.add_argument("--fixed-test-days", default=None, type=lambda s: s.split(","), help="Comma-separated list of fixed test folders.")

    parser.add_argument("--sat-dim", default=4, type=int, help="Number of sat channels.")
    parser.add_argument("--retrieve-dataset", action='store_true', help="Use saved dataset filelist instead of rebuilding.")

    return parser
