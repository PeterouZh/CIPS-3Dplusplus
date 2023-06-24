import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tl2 import tl2_utils
from tl2.proj.fvcore import MODEL_REGISTRY

from .layers import ConvLayer, EqualLinear
from .diffaug import DiffAugment


class ResBlock(nn.Module):
  def __init__(self,
               in_channel,
               out_channel,
               blur_kernel=[1, 3, 3, 1],
               merge=False):
    super().__init__()

    self.conv1 = ConvLayer(2 * in_channel if merge else in_channel, in_channel, 3)
    self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
    self.skip = ConvLayer(2 * in_channel if merge else in_channel, out_channel,
                          1, downsample=True, activate=False, bias=False)
    pass

  def forward(self,
              input):
    out = self.conv1(input)
    out = self.conv2(out)
    out = (out + self.skip(input)) / math.sqrt(2)

    return out


@MODEL_REGISTRY.register(name_prefix=__name__)
class D_StyleGAN(nn.Module):
  def __repr__(self):
    return tl2_utils.get_class_repr(self, __name__)

  def __init__(self,
               input_size,
               channel_multiplier,
               blur_kernel=[1, 3, 3, 1],
               **kwargs):
    super().__init__()

    self.repr_str = tl2_utils.dict2string(dict_obj={
      'input_size': input_size,
      'channel_multiplier': channel_multiplier,
    })
    self.module_name_list = []

    self.input_size = input_size
    self.channel_multiplier = channel_multiplier

    channels = {
      4: 512,
      8: 512,
      16: 512,
      32: 512,
      64: 256 * channel_multiplier,
      128: 128 * channel_multiplier,
      256: 64 * channel_multiplier,
      512: 32 * channel_multiplier,
      1024: 16 * channel_multiplier,
    }

    _in_dim = 3
    _out_dim = channels[input_size]
    convs = [ConvLayer(3, _out_dim, 1)]

    log_size = int(math.log(input_size, 2))

    for i in range(log_size, 2, -1):
      _in_dim = _out_dim
      _out_dim = channels[2 ** (i - 1)]

      _block = ResBlock(_in_dim, _out_dim, blur_kernel)
      convs.append(_block)


    self.convs = nn.Sequential(*convs)

    self.stddev_group = 4
    self.stddev_feat = 1

    # minibatch discrimination
    _in_dim = _out_dim + 1
    _out_dim = channels[4]

    self.final_conv = ConvLayer(_in_dim, _out_dim, 3)
    self.final_linear = nn.Sequential(
      EqualLinear(_out_dim * 4 * 4, _out_dim, activation="fused_lrelu"),
      EqualLinear(_out_dim, 1),
    )

    tl2_utils.print_repr(self)
    pass

  def forward(self,
              input):
    out = self.convs(input)

    # minibatch discrimination
    batch, channel, height, width = out.shape
    group = min(batch, self.stddev_group)
    if batch % group != 0:
      group = 3 if batch % 3 == 0 else 2

    stddev = out.view(
      group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
    )
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    stddev = stddev.repeat(group, 1, height, width)
    final_out = torch.cat([out, stddev], 1)

    # final layers
    final_out = self.final_conv(final_out)
    final_out = final_out.view(batch, -1)
    final_out = self.final_linear(final_out)
    gan_preds = final_out[:, :1]

    return gan_preds


@MODEL_REGISTRY.register(name_prefix=__name__)
class D_StyleGAN_Progressive(nn.Module):
  def __repr__(self):
    return tl2_utils.get_class_repr(self, __name__)

  def __init__(self,
               input_size,
               channel_multiplier,
               pretrained_size=None,
               blur_kernel=[1, 3, 3, 1],
               diffaug=False,
               **kwargs):
    super().__init__()

    self.repr_str = tl2_utils.dict2string(dict_obj={
      'input_size': input_size,
      'channel_multiplier': channel_multiplier,
      'pretrained_size': pretrained_size,
      'diffaug': diffaug,
    })
    self.module_name_list = []

    self.input_size = input_size
    self.channel_multiplier = channel_multiplier
    self.pretrained_size = pretrained_size
    self.diffaug = diffaug

    channels = {
      4: 512,
      8: 512,
      16: 512,
      32: 512,
      64: 256 * channel_multiplier,
      128: 128 * channel_multiplier,
      256: 64 * channel_multiplier,
      512: 32 * channel_multiplier,
      1024: 16 * channel_multiplier,
    }

    self.conv_in = nn.ModuleDict()
    self.blocks = nn.ModuleDict()
    self.module_name_list.extend(['conv_in', 'blocks'])

    for log_size_input in range(10, 2, -1):
      _in_dim = channels[2 ** log_size_input]
      _out_dim = channels[2 ** (log_size_input - 1)]

      _conv_in = ConvLayer(3, _in_dim, 1)
      self.conv_in[f"{2**log_size_input}"] = _conv_in

      _block = ResBlock(_in_dim, _out_dim, blur_kernel)
      self.blocks[f"{2**log_size_input}"] = _block


    self.stddev_group = 4
    self.stddev_feat = 1

    # minibatch discrimination
    _in_dim = channels[4] + 1
    _out_dim = channels[4]

    self.final_conv = ConvLayer(_in_dim, _out_dim, 3)
    self.final_linear = nn.Sequential(
      EqualLinear(_out_dim * 4 * 4, _out_dim, activation="fused_lrelu"),
      EqualLinear(_out_dim, 1),
    )
    self.module_name_list.extend(['final_conv', 'final_linear'])

    tl2_utils.print_repr(self)
    pass

  def diff_aug_img(self, img):
    img = DiffAugment(img, policy='color,translation,cutout')
    return img

  def forward(self,
              input,
              alpha=1.):
    if self.diffaug:
      input = self.diff_aug_img(input)

    B, C, H, W = input.shape
    log_input_size = int(math.log(H, 2))

    if self.pretrained_size is None:
      log_pretrained_size = log_input_size - 1
    elif self.pretrained_size > 0:
      log_pretrained_size = int(math.log(self.pretrained_size, 2))
      if log_pretrained_size == log_input_size:
        log_pretrained_size = log_input_size - 1
    else:
      alpha = 1.
      log_pretrained_size = log_input_size

    x = self.conv_in[f"{2**log_input_size}"](input)

    for log_size_input in range(log_input_size, log_pretrained_size, -1):
      x = self.blocks[f"{2 ** log_size_input}"](x)

    if alpha < 1:
      scale_factor = 2 ** (log_pretrained_size - log_input_size)
      input_downsample = F.interpolate(input, scale_factor=scale_factor, recompute_scale_factor=False,
                                       mode='bilinear', align_corners=False)
      x_downsample = self.conv_in[f"{2**log_pretrained_size}"](input_downsample)

      out = (1 - alpha) * x_downsample + alpha * x
    else:
      out = x

    for log_size_input in range(log_pretrained_size, 2, -1):
      out = self.blocks[f"{2**log_size_input}"](out)

    # minibatch discrimination
    batch, channel, height, width = out.shape
    group = min(batch, self.stddev_group)
    if batch % group != 0:
      group = 3 if batch % 3 == 0 else 2

    stddev = out.view(
      group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
    )
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    stddev = stddev.repeat(group, 1, height, width)
    final_out = torch.cat([out, stddev], 1)

    # final layers
    final_out = self.final_conv(final_out)
    final_out = final_out.view(batch, -1)
    final_out = self.final_linear(final_out)
    gan_preds = final_out[:, :1]

    return gan_preds
