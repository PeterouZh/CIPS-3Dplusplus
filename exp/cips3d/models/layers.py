import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


def make_kernel(k):
  k = torch.tensor(k, dtype=torch.float32)

  if k.ndim == 1:
    k = k[None, :] * k[:, None]

  k /= k.sum()

  return k


class Blur(nn.Module):
  def __init__(self, kernel, pad, upsample_factor=1):
    super().__init__()

    kernel = make_kernel(kernel)

    if upsample_factor > 1:
      kernel = kernel * (upsample_factor ** 2)

    self.register_buffer("kernel", kernel)

    self.pad = pad

  def forward(self, input):
    out = upfirdn2d(input, self.kernel, pad=self.pad)

    return out


class EqualConv2d(nn.Module):
  def __init__(
          self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
  ):
    super().__init__()

    self.weight = nn.Parameter(
      torch.randn(out_channel, in_channel, kernel_size, kernel_size)
    )
    self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

    self.stride = stride
    self.padding = padding

    if bias:
      self.bias = nn.Parameter(torch.zeros(out_channel))

    else:
      self.bias = None

  def forward(self, input):
    out = F.conv2d(
      input,
      self.weight * self.scale,
      bias=self.bias,
      stride=self.stride,
      padding=self.padding,
    )

    return out

  def __repr__(self):
    return (
      f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
      f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
    )


class ConvLayer(nn.Sequential):
  def __init__(self,
               in_channel,
               out_channel,
               kernel_size,
               downsample=False,
               blur_kernel=[1, 3, 3, 1],
               bias=True,
               activate=True):
    layers = []

    if downsample:
      factor = 2
      p = (len(blur_kernel) - factor) + (kernel_size - 1)
      pad0 = (p + 1) // 2
      pad1 = p // 2

      layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

      stride = 2
      self.padding = 0

    else:
      stride = 1
      self.padding = kernel_size // 2

    layers.append(
      EqualConv2d(
        in_channel,
        out_channel,
        kernel_size,
        padding=self.padding,
        stride=stride,
        bias=bias and not activate,
      )
    )

    if activate:
      layers.append(FusedLeakyReLU(out_channel, bias=bias))

    super().__init__(*layers)


class EqualLinear(nn.Module):
  def __init__(self,
               in_dim,
               out_dim,
               bias=True,
               bias_init=0,
               lr_mul=1,
               activation=None):
    super().__init__()

    self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

    if bias:
      self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

    else:
      self.bias = None

    self.activation = activation

    self.scale = (1 / math.sqrt(in_dim)) * lr_mul
    self.lr_mul = lr_mul
    pass

  def forward(self,
              input):
    if self.activation:
      out = F.linear(input, self.weight * self.scale)
      out = fused_leaky_relu(out, self.bias * self.lr_mul)

    else:
      out = F.linear(input, self.weight * self.scale,
                     bias=self.bias * self.lr_mul)

    return out

  def __repr__(self):
    return (
      f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
    )
