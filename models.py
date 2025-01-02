import os
import torch
import torch.nn as nn
import torch.optim as optim
from networks import InpaintGenerator, Discriminator
from src.loss import AdversarialLoss, PerceptualLoss, StyleLoss, GANLoss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_G.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_D.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)

class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        self.lambda_tri = 0.01
        self.lambda_f2b = 1.0
        self.lambda_ff2 = 1.0
        self.domain_distance_loss = nn.TripletMarginLoss(margin=0.1, p=2)
        # self.domain_distance_loss = nn.TripletMarginLoss()
        self.l1_loss = torch.nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002,
                                            betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002,
                                            betas=(0.5, 0.999))
        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

    def process(self, images, masks):

        self.iteration += 1
        self.forward(images, masks)
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        self.backward()
        self.optimizer_G.step()
        self.optimizer_D.step()
        gen_loss = self.gen_loss
        dis_loss = self.dis_loss
        domain_loss = self.domain_loss
        output_stage1 = self.harm1
        output_stage2 = self.harm2
        # create logs
        logs = [
            ("l_gen", dis_loss.item()),
            ("l_dis", gen_loss.item()),
            ("l_tri", domain_loss.item()),
        ]

        return output_stage1, output_stage2, gen_loss, dis_loss, domain_loss, logs

    def forward(self, images, masks):
        self.real = images
        self.mask = masks
        output_stage1, output_stage2, bg_sty_vector, real_fg_sty_vector, coarse_fg_sty_vector, refine_fg_sty_vector = self.generator(self.real, self.mask)
        self.bg_sty_vector, self.real_fg_sty_vector, self.coarse_fg_sty_vector, self.refine_fg_sty_vector = bg_sty_vector, real_fg_sty_vector, coarse_fg_sty_vector, refine_fg_sty_vector
        self.harm1 = output_stage1
        self.harm2 = output_stage2

    def val(self, images, masks):
        output_stage1, output_stage2, bg_sty_vector, real_fg_sty_vector, coarse_fg_sty_vector, refine_fg_sty_vector = self.generator(images, masks)
        return output_stage1, output_stage2, bg_sty_vector, real_fg_sty_vector, coarse_fg_sty_vector, refine_fg_sty_vector

    def backward(self):
        self.domain_loss = (self.domain_distance_loss(self.real_fg_sty_vector, self.refine_fg_sty_vector, self.real_fg_sty_vector)*self.lambda_ff2 +\
            self.domain_distance_loss(self.refine_fg_sty_vector, self.bg_sty_vector, self.refine_fg_sty_vector)*self.lambda_f2b)* self.lambda_tri
        # discriminator loss
        dis_real, _ = self.discriminator(self.real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(self.harm2.detach())                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        self.dis_loss = (dis_real_loss + dis_fake_loss) / 2
        # generator adversarial loss
        gen_input_fake = self.harm2
        gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        self.gen_loss = gen_gan_loss
        # generator l1 loss
        gen_l1_loss = self.l1_loss(self.harm1, self.real) * self.config.L1_LOSS_WEIGHT / torch.mean(self.mask) +\
                        self.l1_loss(self.harm2, self.real) * self.config.L1_LOSS_WEIGHT / torch.mean(self.mask)
        self.gen_loss += gen_l1_loss
        # generator perceptual loss
        gen_perceptual_loss = self.perceptual_loss(self.harm1, self.real) +\
                                self.perceptual_loss(self.harm2, self.real)
        gen_perceptual_loss = gen_perceptual_loss * self.config.PERCEPTUAL_LOSS_WEIGHT
        self.gen_loss += gen_perceptual_loss
        # generator style loss
        gen_style_loss = self.style_loss(self.harm1 * self.mask, self.real * self.mask) + \
                         self.style_loss(self.harm2 * self.mask, self.real * self.mask)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        self.gen_loss += gen_style_loss
        self.loss = self.domain_loss + self.gen_loss + self.dis_loss
        self.loss.backward(retain_graph=True)

