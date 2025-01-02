import os
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from .dataset import Dataset
from models import InpaintingModel
from .utils import Progbar, create_dir, imsave
from .metrics import PSNR


class SafePaint():
    def __init__(self, config):
        self.config = config
        model_name = 'inpaint'
        self.model_name = model_name
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)
        self.psnr = PSNR(255.0).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
            self.results_path = os.path.join(config.RESULTS)
        # train mode
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        self.inpaint_model.load()

    def save(self):
        self.inpaint_model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.inpaint_model.train()

                images, images_gray, masks = self.cuda(*items)
                output_stage1, output_stage2, gen_loss, dis_loss, tri_loss, logs = self.inpaint_model.process(images, masks)
                coarse_imgs = output_stage1 * masks + images * (1 - masks)
                outputs_merged = output_stage2 * masks + images * (1 - masks)
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('gen_loss', gen_loss.item()))
                logs.append(('dis_loss', dis_loss.item()))
                logs.append(('tri_loss', tri_loss.item()))
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                # self.inpaint_model.backward(gen_loss, dis_loss)
                iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

        print('\nEnd training....')

    def test(self):
        self.inpaint_model.eval()
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, masks = self.cuda(*items)
            index += 1
            output_stage1, output_stage2, bg_sty_vector, real_fg_sty_vector, harm_fg_sty_vector, comp_fg_sty_vector = self.inpaint_model.val(images, masks)
            outputs_merged = output_stage2 * masks + images * (1 - masks)
            output = self.postprocess(outputs_merged)[0]
            # path = os.path.join(self.results_path, name)
            path = os.path.join(self.results_path, name.split("/")[-1].replace('.jpg', '.png'))
            print(index, name)
            imsave(output, path)
        print('\nEnd test....')


    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
