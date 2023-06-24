import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tl2.proj.fvcore import MODEL_REGISTRY
from tl2 import tl2_utils

from op import FusedLeakyReLU


class VolumeRenderDiscConv2d(nn.Module):
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               bias=True,
               activate=False):
    super(VolumeRenderDiscConv2d, self).__init__()

    self.conv = nn.Conv2d(in_channels, out_channels,
                          kernel_size, stride, padding, bias=bias and not activate)

    self.activate = activate
    if self.activate:
      self.activation = FusedLeakyReLU(out_channels, bias=bias, scale=1)
      bias_init_coef = np.sqrt(1 / (in_channels * kernel_size * kernel_size))
      nn.init.uniform_(self.activation.bias, a=-bias_init_coef, b=bias_init_coef)
    pass

  def forward(self,
              input):
    """
    input_tensor_shape: (N, C_in,H,W)
    output_tensor_shape: (N,C_out,H_out,W_out）
    :return: Conv2d + activation Result
    """
    out = self.conv(input)
    if self.activate:
      out = self.activation(out)

    return out


class AddCoords(nn.Module):
  def __init__(self):
    super(AddCoords, self).__init__()

  def forward(self, input_tensor):
    """
    :param input_tensor: shape (N, C_in, H, W)
    :return:
    """
    batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
    xx_channel = torch.arange(dim_x, dtype=torch.float32, device=input_tensor.device).repeat(1, 1, dim_y, 1)
    yy_channel = torch.arange(dim_y, dtype=torch.float32, device=input_tensor.device).repeat(1, 1, dim_x, 1).transpose(
      2, 3)

    xx_channel = xx_channel / (dim_x - 1)
    yy_channel = yy_channel / (dim_y - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
    yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)
    out = torch.cat([input_tensor, yy_channel, xx_channel], dim=1)

    return out


class CoordConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
               padding=0, bias=True):
    super(CoordConv2d, self).__init__()

    self.addcoords = AddCoords()
    self.conv = nn.Conv2d(in_channels + 2, out_channels,
                          kernel_size, stride=stride, padding=padding, bias=bias)

  def forward(self, input_tensor):
    """
    input_tensor_shape: (N, C_in,H,W)
    output_tensor_shape: N,C_out,H_out,W_out）
    :return: CoordConv2d Result
    """
    out = self.addcoords(input_tensor)
    out = self.conv(out)

    return out


class CoordConvLayer(nn.Module):
  def __init__(self, in_channel, out_channel, kernel_size, bias=True, activate=True):
    super(CoordConvLayer, self).__init__()
    layers = []
    stride = 1
    self.activate = activate
    self.padding = kernel_size // 2 if kernel_size > 2 else 0

    self.conv = CoordConv2d(in_channel, out_channel, kernel_size,
                            padding=self.padding, stride=stride,
                            bias=bias and not activate)

    if activate:
      self.activation = FusedLeakyReLU(out_channel, bias=bias, scale=1)

    bias_init_coef = np.sqrt(1 / (in_channel * kernel_size * kernel_size))
    nn.init.uniform_(self.activation.bias, a=-bias_init_coef, b=bias_init_coef)

  def forward(self, input):
    out = self.conv(input)
    if self.activate:
      out = self.activation(out)

    return out


class VolumeRenderResBlock(nn.Module):
  def __init__(self, in_channel, out_channel):
    super().__init__()

    self.conv1 = CoordConvLayer(in_channel, out_channel, 3)
    self.conv2 = CoordConvLayer(out_channel, out_channel, 3)
    self.pooling = nn.AvgPool2d(2)
    self.downsample = nn.AvgPool2d(2)
    if out_channel != in_channel:
      self.skip = VolumeRenderDiscConv2d(in_channel, out_channel, 1)
    else:
      self.skip = None

  def forward(self, input):
    out = self.conv1(input)
    out = self.conv2(out)
    out = self.pooling(out)

    downsample_in = self.downsample(input)
    if self.skip != None:
      skip_in = self.skip(downsample_in)
    else:
      skip_in = downsample_in

    out = (out + skip_in) / math.sqrt(2)

    return out


@MODEL_REGISTRY.register(name_prefix=__name__)
class VolumeRenderDiscriminator(nn.Module):
  def __repr__(self):
    return tl2_utils.get_class_repr(self, prefix=__name__)

  def __init__(self,
               input_size,
               viewpoint_loss,
               **kwargs):
    super().__init__()

    self.repr_str = tl2_utils.dict2string(dict_obj={
      'input_size': input_size,
      'viewpoint_loss': viewpoint_loss,
    })
    self.module_name_list = []

    self.input_size = input_size
    self.viewpoint_loss = viewpoint_loss

    final_out_channel = 3 if viewpoint_loss else 1

    channels = {
      2: 400,
      4: 400,
      8: 400,
      16: 400,
      32: 256,
      64: 128,
      128: 64,
    }

    _in_dim = 3
    _out_dim = channels[input_size]
    convs = [VolumeRenderDiscConv2d(3, _out_dim, 1, activate=True)]

    log_size = int(math.log(input_size, 2))

    for i in range(log_size - 1, 0, -1):
      _in_dim = _out_dim
      _out_dim = channels[2 ** i]

      convs.append(VolumeRenderResBlock(_in_dim, _out_dim))

    self.convs = nn.Sequential(*convs)

    _in_dim = _out_dim
    self.final_conv = VolumeRenderDiscConv2d(_in_dim, final_out_channel, 2)

    tl2_utils.print_repr(self)
    pass

  def forward(self,
              input):

    out = self.convs(input)
    out = self.final_conv(out)
    gan_preds = out[:, 0:1]
    gan_preds = gan_preds.view(-1, 1)
    if self.viewpoint_loss:
      viewpoints_preds = out[:, 1:]
      viewpoints_preds = viewpoints_preds.view(-1, 2)
    else:
      viewpoints_preds = None

    return gan_preds, viewpoints_preds


@MODEL_REGISTRY.register(name_prefix=__name__)
class D_VolumeRender_Progressive(nn.Module):
  def __repr__(self):
    return tl2_utils.get_class_repr(self, prefix=__name__)

  def __init__(self,
               input_size,
               viewpoint_loss,
               pretrained_size=None,
               **kwargs):
    super().__init__()

    self.repr_str = tl2_utils.dict2string(dict_obj={
      'input_size': input_size,
      'viewpoint_loss': viewpoint_loss,
      'pretrained_size': pretrained_size,
    })
    self.module_name_list = []

    self.input_size = input_size
    self.viewpoint_loss = viewpoint_loss
    self.pretrained_size = pretrained_size

    final_out_channel = 3 if viewpoint_loss else 1

    channels = {
      2: 400,
      4: 400,
      8: 400,
      16: 400,
      32: 256,
      64: 128,
      128: 64,
      256: 64,
      512: 64,
      1024: 32
    }

    input_size_log = int(math.log(input_size, 2))

    self.conv_in = nn.ModuleDict()
    self.blocks = nn.ModuleDict()
    self.module_name_list.extend(['conv_in', 'blocks'])

    for log_size_input in range(input_size_log, 1, -1):
      _in_dim = channels[2 ** log_size_input]
      _out_dim = channels[2 ** (log_size_input - 1)]

      _conv_in = VolumeRenderDiscConv2d(3, _in_dim, 1, activate=True)
      self.conv_in[f"{2 ** log_size_input}"] = _conv_in

      _block = VolumeRenderResBlock(_in_dim, _out_dim)
      self.blocks[f"{2 ** log_size_input}"] = _block

    _in_dim = channels[2]
    self.final_conv = VolumeRenderDiscConv2d(_in_dim, final_out_channel, 2)
    self.module_name_list.append('final_conv')

    tl2_utils.print_repr(self)
    pass

  def forward(self,
              input,
              alpha=1.):

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

    x = self.conv_in[f"{2 ** log_input_size}"](input)

    for log_size_input in range(log_input_size, log_pretrained_size, -1):
      x = self.blocks[f"{2 ** log_size_input}"](x)

    if alpha < 1:
      scale_factor = 2 ** (log_pretrained_size - log_input_size)
      input_downsample = F.interpolate(input, scale_factor=scale_factor, recompute_scale_factor=False,
                                       mode='bilinear', align_corners=False)
      x_downsample = self.conv_in[f"{2 ** log_pretrained_size}"](input_downsample)

      out = (1 - alpha) * x_downsample + alpha * x
    else:
      out = x

    for log_size_input in range(log_pretrained_size, 1, -1):
      out = self.blocks[f"{2 ** log_size_input}"](out)

    out = self.final_conv(out)
    gan_preds = out[:, 0:1]
    gan_preds = gan_preds.view(-1, 1)
    if self.viewpoint_loss:
      viewpoints_preds = out[:, 1:]
      viewpoints_preds = viewpoints_preds.view(-1, 2)
    else:
      viewpoints_preds = None

    return gan_preds, viewpoints_preds