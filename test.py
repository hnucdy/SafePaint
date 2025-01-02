import os
import cv2
import random
import numpy as np
import torch
import argparse
from src.run import SafePaint


def main(mode=None):

    config = load_config(mode)

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize
    model = SafePaint(config)
    model.load()

    print('\nstart testing...\n')
    model.test()


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--PATH', type=str, default='', help='model checkpoints path')
    parser.add_argument('--TEST_FLIST', type=str, help='path to the input images directory or an input image', default='examples/test/images')
    parser.add_argument('--TEST_MASK_FLIST', type=str, help='path to the masks directory or a mask file', default='examples/test/masks')
    parser.add_argument('--RESULTS', type=str, help='path to the output directory', default='results/test')
    parser.add_argument('--MASK', type=int, help='1: train mode: load mask randomly, 2:test mode: load mask non random', default=2)
    parser.add_argument('--SEED', type=str, help='random seed', default=10)
    parser.add_argument('--GPU', type=str, help='list of gpu ids', default='0')
    parser.add_argument('--INPUT_SIZE', type=int, help='learning rate', default=256)
    parser.add_argument('--GAN_LOSS', type=str, help='', default='nsgan')
    args = parser.parse_args()
    # 1: train 2: test
    args.MODE = 2
    return args


if __name__ == "__main__":
    main()
