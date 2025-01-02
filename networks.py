import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
import numbers
import math

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class InpaintGenerator(nn.Module):
    def __init__(self, use_cuda=True, device_ids=None):
        super(InpaintGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.coarse_generator = CoarseGenerator()
        self.fine_generator = FineGenerator()
        self.extractor = StyleEncoder(style_dim=16, norm_layer=nn.BatchNorm2d)

    def forward(self, images, masks):
        # stage 1
        images_masked_stage1 = images * (1 - masks) + masks
        inputs_stage1 = torch.cat((images_masked_stage1, masks), dim=1)
        output_stage1 = self.coarse_generator(inputs_stage1)
        outputs_merged_stage1 = output_stage1 * masks + images * (1 - masks)
        # stage 2
        bg_sty_vector = self.extractor(images, 1 - masks)
        bg_sty_map = bg_sty_vector.expand([bg_sty_vector.size(0), 16, 256, 256])
        inputs_stage2 = torch.cat((outputs_merged_stage1, masks, bg_sty_map), dim=1)
        output_stage2 = self.fine_generator(inputs_stage2, masks)
        outputs_merged_stage2 = images * (1 - masks) + output_stage2 * masks
        real_fg_sty_vector = self.extractor(images, masks)
        coarse_fg_sty_vector = self.extractor(outputs_merged_stage1, masks)
        refine_fg_sty_vector = self.extractor(outputs_merged_stage2, masks)
        return output_stage1, output_stage2, bg_sty_vector, real_fg_sty_vector, coarse_fg_sty_vector, refine_fg_sty_vector

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        self.return_mask = True

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output

class PartialGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(PartialGenerator, self).__init__()

        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = PartialConv2d(3, 64, kernel_size=7, stride=1, padding=0)
        self.norm1 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.relu1 = nn.ReLU(True)

        self.conv2 = PartialConv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(128, track_running_stats=False)
        self.relu2 = nn.ReLU(True)

        self.conv3 = PartialConv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(256, track_running_stats=False)
        self.relu3 = nn.ReLU(True)

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.conv4 = PartialConv2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm2d(128, track_running_stats=False)
        self.relu4 = nn.ReLU(True)

        self.conv5 = PartialConv2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.norm5 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.relu5 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(3)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)

        if init_weights:
            self.init_weights()

        self.middle = nn.Sequential(*blocks)


    def forward(self, x, m):
        # x = self.pad1(x)
        x1, m1 = self.conv1(x, m)
        x1 = self.norm1(x1)
        x1 = self.relu1(x1)
        x2, m2 = self.conv2(x1, m1)
        x2 = self.norm2(x2)
        x2 = self.relu2(x2)
        x3, m3 = self.conv3(x2, m2)
        x3 = self.norm3(x3)
        x3 = self.relu3(x3)
        x3 = self.middle(x3)
        x4, m4 = self.conv4(x3, m3)
        x4 = self.norm4(x4)
        x4 = self.relu4(x4)
        x5, m5 = self.conv5(x4, m4)
        x5 = self.norm5(x5)
        x5 = self.relu5(x5)
        # x5 = self.pad2(x5)
        x6 = self.conv6(x5)
        x6 = (torch.tanh(x6) + 1) / 2
        return x6

class CoarseGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(CoarseGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x

class FineGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(FineGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=20, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)
        self.att1 = RASC(256)
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm2d(128, track_running_stats=False)
        self.relu1 = nn.ReLU(True)

        self.att2 = RASC(128)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.relu2 = nn.ReLU(True)

        self.att3 = RASC(64)
        self.pad1 = nn.ReflectionPad2d(3)
        self.deconv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)

        if init_weights:
            self.init_weights()

    def forward(self, x, masks):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.att1(x, masks)
        x = self.deconv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.att2(x, masks)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.att3(x, masks)
        x = self.pad1(x)
        x = self.deconv3(x)
        x = (torch.tanh(x) + 1) / 2

        return x

class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

class StyleEncoder(BaseNetwork):
    def __init__(self, style_dim, norm_layer=nn.BatchNorm2d, init_weights=True):
        super(StyleEncoder, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        ndf=64
        n_layers=6
        kw = 3
        padw = 0
        self.conv1f = PartialConv2d(3, ndf, kernel_size=kw, stride=2, padding=padw)
        self.relu1 = nn.ReLU(True)
        nf_mult = 1
        nf_mult_prev = 1

        n = 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv2f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm2f = norm_layer(ndf * nf_mult)
        self.relu2 = nn.ReLU(True)

        n = 2
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv3f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm3f = norm_layer(ndf * nf_mult)
        self.relu3 = nn.ReLU(True)

        n = 3
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv4f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm4f = norm_layer(ndf * nf_mult)
        self.relu4 = nn.ReLU(True)

        n = 4
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        self.conv5f = PartialConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.convs = nn.Conv2d(ndf * nf_mult, style_dim, kernel_size=1, stride=1)

        if init_weights:
            self.init_weights()

    def forward(self, input, mask):
        """Standard forward."""
        xb = input
        mb = mask

        xb, mb = self.conv1f(xb, mb)
        xb = self.relu1(xb)
        xb, mb = self.conv2f(xb, mb)
        xb = self.norm2f(xb)
        xb = self.relu2(xb)
        xb, mb = self.conv3f(xb, mb)
        xb = self.norm3f(xb)
        xb = self.relu3(xb)
        xb, mb = self.conv4f(xb, mb)
        xb = self.norm4f(xb)
        xb = self.relu4(xb)
        xb, mb = self.conv5f(xb, mb)
        xb = self.avg_pooling(xb)
        s = self.convs(xb)
        return s

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class GlobalAttentionModule(nn.Module):
    """docstring for GlobalAttentionModule"""

    def __init__(self, channel, reducation=16):
        super(GlobalAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, channel // reducation),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reducation, channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, w, h = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y2 = self.max_pool(x).view(b, c)
        y = self.fc(torch.cat([y1, y2], 1)).view(b, c, 1, 1)
        return x * y

class BasicLearningBlock(nn.Module):
    """docstring for BasicLearningBlock"""

    def __init__(self, channel):
        super(BasicLearningBlock, self).__init__()
        self.rconv1 = nn.Conv2d(channel, channel * 2, 3, padding=1, bias=False)
        self.rbn1 = nn.BatchNorm2d(channel * 2)
        self.rconv2 = nn.Conv2d(channel * 2, channel, 3, padding=1, bias=False)
        self.rbn2 = nn.BatchNorm2d(channel)

    def forward(self, feature):
        return F.elu(self.rbn2(self.rconv2(F.elu(self.rbn1(self.rconv1(feature))))))

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

class RASC(nn.Module):
    def __init__(self,channel,type_of_connection=BasicLearningBlock):
        super(RASC, self).__init__()
        self.connection = type_of_connection(channel)
        self.background_attention = GlobalAttentionModule(channel,16)
        self.mixed_attention = GlobalAttentionModule(channel,16)
        self.spliced_attention = GlobalAttentionModule(channel,16)
        self.gaussianMask = GaussianSmoothing(1,5,1)
        self.fuser = nn.Conv2d(channel * 2, channel, kernel_size=1)

    def forward(self,feature,mask):
        _,_,w,_ = feature.size()
        _,_,mw,_ = mask.size()
        # binaryfiy
        # selected the feature from the background as the additional feature to masked splicing feature.
        if w != mw:
            mask = torch.round(F.avg_pool2d(mask,2,stride=mw//w))
        reverse_mask = -1*(mask-1)
        # here we add gaussin filter to mask and reverse_mask for better harimoization of edges.

        mask = self.gaussianMask(F.pad(mask,(2,2,2,2),mode='reflect'))
        reverse_mask = self.gaussianMask(F.pad(reverse_mask,(2,2,2,2),mode='reflect'))


        background = self.background_attention(feature) * reverse_mask
        selected_feature = self.mixed_attention(feature)
        spliced_feature = self.spliced_attention(feature)
        spliced = (self.connection(spliced_feature) + selected_feature) * mask
        outputs = self.fuser(torch.cat([background + spliced, feature],dim = 1))
        return outputs