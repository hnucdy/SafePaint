import os
import cv2
import random
import numpy as np
import torch
import argparse
from src.run import SafePaint


def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

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

    print('\nstart training...\n')
    model.train()


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--PATH', type=str, default='', help='model checkpoints path')
    parser.add_argument('--TRAIN_FLIST', type=str, help='path to the input images directory or an input image', default='')
    parser.add_argument('--TRAIN_MASK_FLIST', type=str, help='path to the masks directory or a mask file', default='')
    parser.add_argument('--MASK', type=int, help='1: train mode: load mask randomly, 2:test mode: load mask non random', default=1)

    parser.add_argument('--SEED', type=str, help='random seed', default=10)
    parser.add_argument('--GPU', type=str, help='list of gpu ids', default='0')
    parser.add_argument('--VERBOSE', type=int, help='turns on verbose mode in the output console', default='0')

    parser.add_argument('--LR', type=float, help='learning rate', default=0.0001)
    parser.add_argument('--D2G_LR', type=float, help='discriminator/generator learning rate ratio', default=0.1)
    parser.add_argument('--BETA1', type=float, help='adam optimizer beta1', default=0.0)
    parser.add_argument('--BETA2', type=float, help='adam optimizer beta2', default=0.9)
    parser.add_argument('--BATCH_SIZE', type=int, help='input batch size for training', default=4)
    parser.add_argument('--INPUT_SIZE', type=int, help='learning rate', default=256)
    parser.add_argument('--MAX_ITERS', type=float, help='maximum number of iterations to train the model', default=1000000)
    parser.add_argument('--SAVE_INTERVAL', type=int, help='how many iterations to wait before saving model (0: never)', default=1000)
    parser.add_argument('--LOG_INTERVAL', type=int, help='how many iterations to wait before logging training status (0: never)', default=10)

    parser.add_argument('--L1_LOSS_WEIGHT', type=float, help='l1 loss weight', default=1)
    parser.add_argument('--FM_LOSS_WEIGHT', type=float, help='feature-matching loss weight', default=10)
    parser.add_argument('--STYLE_LOSS_WEIGHT', type=float, help='style loss weight', default=250)
    parser.add_argument('--PERCEPTUAL_LOSS_WEIGHT', type=float, help='perceptual loss weight', default=0.1)
    parser.add_argument('--INPAINT_ADV_LOSS_WEIGHT', type=float, help='adversarial loss weight', default=0.1)
    parser.add_argument('--GAN_LOSS', type=str, help='', default='nsgan')


    args = parser.parse_args()
    # 1: train 2: test
    args.MODE = 1
    return args


if __name__ == "__main__":
    main()
