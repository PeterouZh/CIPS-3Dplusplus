import math
import random
import trimesh
import numpy as np
from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as F

from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
from pytorch3d.transforms import matrix_to_euler_angles

from tl2 import tl2_utils
from tl2.proj.fvcore import MODEL_REGISTRY
from tl2.proj.pytorch import torch_utils

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

from exp.cips3d.models.volume_renderer_v5 import VolumeFeatureRenderer
from exp.stylesdf.utils import (
  create_cameras,
  create_mesh_renderer,
  add_textures,
  create_depth_mesh_renderer,
)
from exp.cips3d import nerf_utils


class PixelNorm(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, input):
    return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class MappingLinear(nn.Module):
  def __init__(self, in_dim, out_dim, bias=True, activation=None, is_last=False):
    super().__init__()
    if is_last:
      weight_std = 0.25
    else:
      weight_std = 1

    self.weight = nn.Parameter(weight_std * nn.init.kaiming_normal_(torch.empty(out_dim, in_dim), a=0.2, mode='fan_in',
                                                                    nonlinearity='leaky_relu'))

    if bias:
      self.bias = nn.Parameter(nn.init.uniform_(torch.empty(out_dim), a=-np.sqrt(1 / in_dim), b=np.sqrt(1 / in_dim)))
    else:
      self.bias = None

    self.activation = activation

  def forward(self, input):
    if self.activation != None:
      out = F.linear(input, self.weight)
      out = fused_leaky_relu(out, self.bias, scale=1)
    else:
      out = F.linear(input, self.weight, bias=self.bias)

    return out

  def __repr__(self):
    return (
      f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
    )


def make_kernel(k):
  k = torch.tensor(k, dtype=torch.float32)

  if k.ndim == 1:
    k = k[None, :] * k[:, None]

  k /= k.sum()

  return k


class Upsample(nn.Module):
  def __init__(self, kernel, factor=2):
    super().__init__()

    self.factor = factor
    kernel = make_kernel(kernel) * (factor ** 2)
    self.register_buffer("kernel", kernel)

    p = kernel.shape[0] - factor

    pad0 = (p + 1) // 2 + factor - 1
    pad1 = p // 2

    self.pad = (pad0, pad1)

  def forward(self, input):
    out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

    return out


class Downsample(nn.Module):
  def __init__(self, kernel, factor=2):
    super().__init__()

    self.factor = factor
    kernel = make_kernel(kernel)
    self.register_buffer("kernel", kernel)

    p = kernel.shape[0] - factor

    pad0 = (p + 1) // 2
    pad1 = p // 2

    self.pad = (pad0, pad1)

  def forward(self, input):
    out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

    return out


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


class EqualLinear(nn.Module):
  def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1,
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

  def forward(self, input):
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


def get_symm_conv_kernels(weight_raw):
  """
  
  :param weight_raw: (out, in, rank_dim)
  :return
  
  - weights: (out, in, ks, ks)
  
  """
  # weight_raw = F.normalize(weight_raw, dim=-1)

  weight_raw_flip = weight_raw.flip([2])
  weight_col = torch.cat([weight_raw, weight_raw_flip[..., 1:]], dim=-1).unsqueeze(-1)
  weight_row = weight_col.clone()
  weights = torch.einsum('abik,abjk->abij', weight_col, weight_row)
  
  return weights


def get_symm_conv_kernels_3x3(xx,
                              xy,
                              yy):
  """

  :param xx: (out, in, 1)
  :param xy:
  :param yy:
  :return:
  """
  out_channel, in_channel, _ = xx.shape

  xx = xx * 0.25
  xy = xy * 0.25
  weight = torch.cat([xx, xy, xx,
                      xy, yy, xy,
                      xx, xy, xx], dim=-1).reshape(out_channel, in_channel, 3, 3)
  return weight

class ModulatedConv2d(nn.Module):
  def __init__(self,
               in_channel,
               out_channel,
               kernel_size,
               style_dim,
               demodulate=True,
               upsample=False,
               downsample=False,
               blur_kernel=[1, 3, 3, 1],
               use_symm_conv=False):
    super().__init__()

    self.eps = 1e-8
    self.kernel_size = kernel_size
    self.in_channel = in_channel
    self.out_channel = out_channel
    self.upsample = upsample
    self.downsample = downsample
    self.use_symm_conv = use_symm_conv and kernel_size > 1

    if upsample:
      factor = 2
      p = (len(blur_kernel) - factor) - (kernel_size - 1)
      pad0 = (p + 1) // 2 + factor - 1
      pad1 = p // 2 + 1

      self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

    if downsample:
      factor = 2
      p = (len(blur_kernel) - factor) + (kernel_size - 1)
      pad0 = (p + 1) // 2
      pad1 = p // 2

      self.blur = Blur(blur_kernel, pad=(pad0, pad1))

    fan_in = in_channel * kernel_size ** 2
    self.scale = 1 / math.sqrt(fan_in)
    self.padding = kernel_size // 2

    if self.use_symm_conv:
      if kernel_size == 3:
        self.weight_xx = nn.Parameter(torch.randn(out_channel, in_channel, 1))
        self.weight_xy = nn.Parameter(torch.randn(out_channel, in_channel, 1))
        self.weight_yy = nn.Parameter(torch.randn(out_channel, in_channel, 1))

      else:
        raise NotImplementedError
      
    else:
      self.weight = nn.Parameter(
        torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
      )

    self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

    self.demodulate = demodulate

  def __repr__(self):
    return (
      f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
      f"upsample={self.upsample}, downsample={self.downsample})"
    )

  def forward(self,
              input,
              style):
    batch, in_channel, height, width = input.shape
    style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
    
    if self.use_symm_conv:
      if self.kernel_size == 3:
        weight = get_symm_conv_kernels_3x3(xx=self.weight_xx,
                                           xy=self.weight_xy,
                                           yy=self.weight_yy)
        weight = weight.unsqueeze(0)

        weight = self.scale * weight * style
      else:
        raise NotImplementedError
      
    else:
      weight = self.scale * self.weight * style

    if self.demodulate:
      demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
      weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

    weight = weight.view(
      batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
    )

    if self.upsample:
      input = input.view(1, batch * in_channel, height, width)
      weight = weight.view(
        batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
      )
      weight = weight.transpose(1, 2).reshape(
        batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
      )
      out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
      _, _, height, width = out.shape
      out = out.view(batch, self.out_channel, height, width)
      out = self.blur(out)

    elif self.downsample:
      input = self.blur(input)
      _, _, height, width = input.shape
      input = input.view(1, batch * in_channel, height, width)
      out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
      _, _, height, width = out.shape
      out = out.view(batch, self.out_channel, height, width)

    else:
      if height == 1 or width == 1:
        input = rearrange(input, "b c h w -> b (h w) c")
        weight = rearrange(weight, "(b cout) cin 1 1 -> b cin cout", b=batch)
        out = torch.bmm(input, weight)
        out = rearrange(out, "b (h w) c -> b c h w", h=height, w=width)

      else:
        input = input.reshape(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

    return out


class NoiseInjection(nn.Module):
  def __init__(self, project=False):
    super().__init__()
    self.project = project
    self.weight = nn.Parameter(torch.zeros(1))
    self.prev_noise = None
    self.mesh_fn = None
    self.vert_noise = None
    pass

  def forward(self,
              image,
              noise=None,
              transform=None,
              mesh_path=None):
    batch, _, height, width = image.shape
    if noise is None:
      noise = image.new_empty(batch, 1, height, width).normal_()
      return image + self.weight * noise
      # return image
    elif self.project:
      noise = self.project_noise(noise, transform, mesh_path=mesh_path)
      return image + self.weight * noise
    else:
      return image + self.weight * noise


  def create_pytorch_mesh(self, trimesh):
    v = trimesh.vertices
    f = trimesh.faces
    verts = torch.from_numpy(np.asarray(v)).to(torch.float32).cuda()
    mesh_pytorch = Meshes(
      verts=[verts],
      faces=[torch.from_numpy(np.asarray(f)).to(torch.float32).cuda()],
      textures=None
    )
    if self.vert_noise == None or verts.shape[0] != self.vert_noise.shape[1]:
      self.vert_noise = torch.ones_like(verts)[:, 0:1].cpu().normal_().expand(-1, 3).unsqueeze(0)

    mesh_pytorch = add_textures(meshes=mesh_pytorch, vertex_colors=self.vert_noise.to(verts.device))

    return mesh_pytorch

  def load_mc_mesh(self, filename, resolution=128, im_res=64):
    import trimesh

    mc_tri = trimesh.load_mesh(filename)
    v = mc_tri.vertices;
    f = mc_tri.faces
    mesh2 = trimesh.base.Trimesh(vertices=v, faces=f)
    if im_res == 64 or im_res == 128:
      pytorch3d_mesh = self.create_pytorch_mesh(mesh2)
      return pytorch3d_mesh
    v, f = trimesh.remesh.subdivide(v, f)
    mesh2_subdiv = trimesh.base.Trimesh(vertices=v, faces=f)
    if im_res == 256:
      pytorch3d_mesh = self.create_pytorch_mesh(mesh2_subdiv);
      return pytorch3d_mesh
    v, f = trimesh.remesh.subdivide(mesh2_subdiv.vertices, mesh2_subdiv.faces)
    mesh3_subdiv = trimesh.base.Trimesh(vertices=v, faces=f)
    if im_res == 256:
      pytorch3d_mesh = self.create_pytorch_mesh(mesh3_subdiv);
      return pytorch3d_mesh
    v, f = trimesh.remesh.subdivide(mesh3_subdiv.vertices, mesh3_subdiv.faces)
    mesh4_subdiv = trimesh.base.Trimesh(vertices=v, faces=f)

    pytorch3d_mesh = self.create_pytorch_mesh(mesh4_subdiv)

    return pytorch3d_mesh

  def project_noise(self, noise, transform, mesh_path=None):
    batch, _, height, width = noise.shape
    assert (batch == 1)  # assuming during inference batch size is 1

    angles = matrix_to_euler_angles(transform[0:1, :, :3], "ZYX")
    azim = float(angles[0][1])
    elev = float(-angles[0][2])

    cameras = create_cameras(azim=azim * 180 / np.pi, elev=elev * 180 / np.pi, fov=12., dist=1)

    renderer = create_depth_mesh_renderer(cameras, image_size=height,
                                          specular_color=((0, 0, 0),), ambient_color=((1., 1., 1.),),
                                          diffuse_color=((0, 0, 0),))

    if self.mesh_fn is None or self.mesh_fn != mesh_path:
      self.mesh_fn = mesh_path

    pytorch3d_mesh = self.load_mc_mesh(mesh_path, im_res=height)
    rgb, depth = renderer(pytorch3d_mesh)

    depth_max = depth.max(-1)[0].view(-1)  # (NxN)
    depth_valid = depth_max > 0.
    if self.prev_noise is None:
      self.prev_noise = noise
    noise_copy = self.prev_noise.clone()
    noise_copy.view(-1)[depth_valid] = rgb[0, :, :, 0].view(-1)[depth_valid]
    noise_copy = noise_copy.reshape(1, 1, height, height)  # 1x1xNxN

    return noise_copy


class StyledConv(nn.Module):
  def __init__(self,
               in_channel,
               out_channel,
               kernel_size,
               style_dim,
               upsample=False,
               blur_kernel=[1, 3, 3, 1],
               project_noise=False,
               use_symm_conv=False):
    super().__init__()

    self.conv = ModulatedConv2d(
      in_channel,
      out_channel,
      kernel_size,
      style_dim,
      upsample=upsample,
      blur_kernel=blur_kernel,
      use_symm_conv=use_symm_conv,
    )

    self.noise = NoiseInjection(project=project_noise)

    self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
    self.activate = FusedLeakyReLU(out_channel)
    pass

  def forward(self,
              input,
              style,
              noise=None,
              transform=None,
              mesh_path=None):
    out = self.conv(input, style)
    out = self.noise(out, noise=noise, transform=transform, mesh_path=mesh_path)
    out = self.activate(out)

    return out


class ToRGB(nn.Module):
  def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
    super().__init__()

    self.upsample = upsample
    out_channels = 3
    if upsample:
      self.upsample = Upsample(blur_kernel)

    self.conv = ModulatedConv2d(in_channel, out_channels, 1, style_dim, demodulate=False)
    self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

  def forward(self,
              input,
              style,
              skip=None):
    out = self.conv(input, style)
    out = out + self.bias

    if skip is not None:
      if self.upsample:
        skip = self.upsample(skip)

      out = out + skip

    return out


class ConvLayer(nn.Sequential):
  def __init__(self, in_channel, out_channel, kernel_size, downsample=False,
               blur_kernel=[1, 3, 3, 1], bias=True, activate=True):
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


class Decoder(nn.Module):
  def __repr__(self):
    return tl2_utils.get_class_repr(self, prefix=__name__)

  def __init__(self,
               size_start,
               size_end,
               style_dim,
               in_channel,
               channel_multiplier,
               project_noise,
               upsample_list=[],
               kernel_size=1,
               blur_kernel=[1, 3, 3, 1],
               use_symm_conv=False,
               **kwargs):
    super().__init__()

    self.repr_str = tl2_utils.dict2string(dict_obj={
      'size_start': size_start,
      'size_end': size_end,
      'style_dim': style_dim,
      'in_channel': in_channel,
      'channel_multiplier': channel_multiplier,
      'project_noise': project_noise,
      'upsample_list': upsample_list,
      'kernel_size': kernel_size,
      'use_symm_conv': use_symm_conv,
    })
    self.module_name_list = []

    # decoder mapping network
    self.size_start = size_start
    self.size_end = size_end
    # self.z_dim = z_dim
    self.style_dim = style_dim
    # self.lr_mul_mapping = lr_mul_mapping
    self.in_channel = in_channel
    self.channel_multiplier = channel_multiplier
    self.project_noise = project_noise
    self.upsample_list = upsample_list
    self.kernel_size = kernel_size
    self.blur_kernel = blur_kernel
    self.use_symm_conv = use_symm_conv

    self.channels = {
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
    # self.channels = {
    #   4: 512,
    #   8: 512,
    #   16: 512,
    #   32: 512,
    #   64: 512,
    #   128: 512,
    #   256: 512,
    #   512: 512,
    #   1024: 512,
    # }

    self.create_synthesis()

    tl2_utils.print_repr(self)
    pass

  def forward(self,
              features,
              styles,
              rgbd_in=None,
              transform=None,
              noise=None,
              mesh_path=None):

    latent = styles

    out = self.conv1(input=features,
                     style=latent[:, 0],
                     noise=noise[0],
                     transform=transform,
                     mesh_path=mesh_path)

    skip = self.to_rgb1(input=out,
                        style=latent[:, 1],
                        skip=rgbd_in)

    i = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(self.convs[::2],
                                                    self.convs[1::2],
                                                    noise[1::2],
                                                    noise[2::2],
                                                    self.to_rgbs):
      out = conv1(input=out,
                  style=latent[:, i],
                  noise=noise1,
                  transform=transform,
                  mesh_path=mesh_path)
      out = conv2(input=out,
                  style=latent[:, i + 1],
                  noise=noise2,
                  transform=transform,
                  mesh_path=mesh_path)
      skip = to_rgb(input=out,
                    style=latent[:, i + 2],
                    skip=skip)

      i += 2

    # out_latent = latent if return_latents else None
    image = skip

    return image

  def create_noise_bufs(self,
                        start_size,
                        device):
    # noise bufs

    noise_bufs = []
    shape = [1, 1, start_size, start_size]
    noise_bufs.append(torch.randn(*shape, device=device))

    cur_size = start_size

    for i in range(self.log_in_size + 1, self.log_size + 1):  # [3, 11)

      if 2 ** i in self.upsample_list:
        cur_size *= 2

      shape = [1, 1, cur_size, cur_size]
      noise_bufs.append(torch.randn(*shape, device=device))
      noise_bufs.append(torch.randn(*shape, device=device))

    # N_bufs = (self.log_size - self.log_in_size) * 2 + 1
    # for layer_idx in range(N_bufs):
    #   res = (layer_idx + 2 * self.log_in_size + 1) // 2
    #   shape = [1, 1, 2 ** (res), 2 ** (res)]
    #   # self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))
    #   noise_bufs.append(torch.randn(*shape, device=device))

    return noise_bufs

  def create_synthesis(self):

    # image decoder
    self.log_in_size = int(math.log(self.size_start, 2)) # 2
    self.log_size = int(math.log(self.size_end, 2)) # 10

    _in_dim = self.in_channel
    _out_dim = self.channels[self.size_start]

    self.conv1 = StyledConv(in_channel=_in_dim,
                            out_channel=_out_dim,
                            kernel_size=self.kernel_size,
                            style_dim=self.style_dim,
                            blur_kernel=self.blur_kernel,
                            project_noise=self.project_noise,
                            use_symm_conv=self.use_symm_conv)
    self.to_rgb1 = ToRGB(_out_dim, self.style_dim, upsample=False)
    self.module_name_list.extend(['conv1', 'to_rgb1'])

    self.convs = nn.ModuleList()
    # self.upsamples = nn.ModuleList()
    self.to_rgbs = nn.ModuleList()
    self.noises = nn.Module()
    self.module_name_list.extend(['convs', 'to_rgbs', 'noises'])

    # for layer_idx in range(self.num_layers):
    #   res = (layer_idx + 2 * self.log_in_size + 1) // 2
    #   shape = [1, 1, 2 ** (res), 2 ** (res)]
    #   self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

    for i in range(self.log_in_size + 1, self.log_size + 1): # [3, 11)
      _in_dim = _out_dim
      _out_dim = self.channels[2 ** i]

      if 2 ** i in self.upsample_list:
        upsample = True
      else:
        upsample = False

      self.convs.append(StyledConv(in_channel=_in_dim,
                                   out_channel=_out_dim,
                                   kernel_size=self.kernel_size,
                                   style_dim=self.style_dim,
                                   upsample=upsample,
                                   blur_kernel=self.blur_kernel,
                                   project_noise=self.project_noise,
                                   use_symm_conv=self.use_symm_conv))

      self.convs.append(StyledConv(in_channel=_out_dim,
                                   out_channel=_out_dim,
                                   kernel_size=self.kernel_size,
                                   style_dim=self.style_dim,
                                   blur_kernel=self.blur_kernel,
                                   project_noise=self.project_noise,
                                   use_symm_conv=self.use_symm_conv))

      self.to_rgbs.append(ToRGB(in_channel=_out_dim,
                                style_dim=self.style_dim,
                                upsample=upsample))

    # N_conv_layers (10 - 2) * 2 + 1
    self.num_layers = (self.log_size - self.log_in_size) * 2 + 1
    # N_styles
    self.n_latent = (self.log_size - self.log_in_size) * 2 + 2
    pass

  def create_mapping(self,
                     z_dim,
                     style_dim,
                     lr_mul_mapping):
    layers = [
      PixelNorm(),
      EqualLinear(z_dim, style_dim, lr_mul=lr_mul_mapping, activation="fused_lrelu"),
    ]
    self.module_name_list.extend(['style.0', 'style.1'])

    for i in range(4):
      layers.append(
        EqualLinear(style_dim, style_dim, lr_mul=lr_mul_mapping, activation="fused_lrelu")
      )
      self.module_name_list.append(f'style.{i + 2}')

    self.style = nn.Sequential(*layers)
    self.module_name_list.append('style')
    pass

  def mean_latent(self, renderer_latent):
    latent = self.style(renderer_latent).mean(0, keepdim=True)

    return latent

  def get_latent(self, input):
    return self.style(input)

  # def styles_and_noise_forward(self,
  #                              styles,
  #                              noise,
  #                              inject_index=None,
  #                              truncation=1,
  #                              truncation_latent=None,
  #                              input_is_latent=False,
  #                              randomize_noise=True):
  #   if not input_is_latent:
  #     styles = [self.style(s) for s in styles]
  #
  #   if noise is None:
  #     if randomize_noise:
  #       noise = [None] * self.num_layers
  #     else:
  #       noise = [
  #         getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
  #       ]
  #
  #   if (truncation < 1):
  #     style_t = []
  #
  #     for style in styles:
  #       style_t.append(
  #         truncation_latent[1] + truncation * (style - truncation_latent[1])
  #       )
  #
  #     styles = style_t
  #
  #   if len(styles) < 2:
  #     inject_index = self.n_latent
  #
  #     if styles[0].ndim < 3:
  #       latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
  #
  #     else:
  #       latent = styles[0]
  #   else:
  #     if inject_index is None:
  #       inject_index = random.randint(1, self.n_latent - 1)
  #
  #     latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
  #     latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
  #
  #     latent = torch.cat([latent, latent2], 1)
  #
  #   return latent, noise


class Generator_utils(object):
  
  def get_noise_bufs(self,
                     noise_bufs,
                     zero_noise):
    if noise_bufs is None:
      N_conv_layers = self.decoder.num_layers

      if zero_noise:
        noise_bufs = [0] * N_conv_layers
      else:  # random noise
        noise_bufs = [None] * N_conv_layers
    
    return noise_bufs

@MODEL_REGISTRY.register(name_prefix=__name__)
class Generator(nn.Module, Generator_utils):
  def __repr__(self):
    return tl2_utils.get_class_repr(self, prefix=__name__)

  def __init__(self,
               enable_decoder,
               freeze_renderer,
               renderer_detach=True,
               predict_rgb_residual=False,
               scale_factor=None,
               renderer_cfg={},
               mapping_renderer_cfg={},
               decoder_cfg={},
               mapping_decoder_cfg={},
               **kwargs):
    super().__init__()

    self.repr_str = tl2_utils.dict2string(dict_obj={
      'enable_decoder': enable_decoder,
      'freeze_renderer': freeze_renderer,
      'renderer_detach': renderer_detach,
      'predict_rgb_residual': predict_rgb_residual,
      'scale_factor': scale_factor,
    })

    self.enable_decoder = enable_decoder
    self.freeze_renderer = freeze_renderer
    self.renderer_detach = renderer_detach
    self.predict_rgb_residual = predict_rgb_residual
    self.scale_factor = scale_factor

    self.renderer_cfg = renderer_cfg
    self.mapping_renderer_cfg = mapping_renderer_cfg
    self.decoder_cfg = decoder_cfg
    self.mapping_decoder_cfg = mapping_decoder_cfg

    self.module_name_list = []

    # nerf net
    self.renderer = VolumeFeatureRenderer(style_dim=mapping_renderer_cfg['style_dim'],
                                          **renderer_cfg)
    self.module_name_list.append('renderer')
    self.N_layers_renderer = self.renderer.N_layers_renderer

    self.create_mapping_nerf(**mapping_renderer_cfg)
    self.z_dim = mapping_renderer_cfg['z_dim']

    # fc net
    self.decoder = Decoder(style_dim=mapping_decoder_cfg['style_dim'],
                           **{**decoder_cfg,
                              'in_channel': renderer_cfg['hidden_dim']})
    self.module_name_list.append('decoder')

    self.create_mapping_decoder(z_dim=mapping_renderer_cfg['style_dim'],
                                **mapping_decoder_cfg)

    # if scale_factor > 1:
    #   # interp_mode = 'bilinear'
    #   interp_mode = 'bicubic'
    #   self.upsample_layer = nn.Upsample(scale_factor=scale_factor, mode=interp_mode, align_corners=False)
    # else:
    #   self.upsample_layer = None

    tl2_utils.print_repr(self)
    pass

  def forward(self,
              zs, # [(b, style_dim)]
              # camera kwargs
              cam_poses,
              focals,
              img_size,
              near=0.88,
              far=1.12,
              # mapping kwargs
              truncation=1,
              inject_index=None,
              path_reg=False,
              style_render=None,
              style_decoder=None,
              # noise bufs
              noise_bufs=None,
              zero_noise=True,
              # aux
              eikonal_reg=False,
              return_sdf=False,
              return_xyz=False,
              N_rays_forward=None, # grad or no_grad
              N_rays_grad=None, # grad
              N_samples_forward=None, # no_grad
              nerf_cfg={},
              recompute_mean=False,
              # others
              project_noise=False,
              mesh_path=None,
              renderer_detach=None,
              enable_decoder=None,
              **kwargs):
    if renderer_detach is None:
      renderer_detach = self.renderer_detach
    if enable_decoder is None:
      enable_decoder = self.enable_decoder

    if eikonal_reg:
      if N_rays_grad is None:
        assert N_rays_forward is None
    assert len(zs) == 2

    # do not calculate renderer gradients if renderer weights are frozen
    if self.freeze_renderer:
      self.style.requires_grad_(False)
      self.renderer.requires_grad_(False)

    style_render, style_decoder = self.mapping_networks(zs=zs,
                                                        truncation=truncation,
                                                        inject_index=inject_index,
                                                        path_reg=path_reg,
                                                        style_render=style_render,
                                                        style_decoder=style_decoder,
                                                        recompute_mean=recompute_mean)
    
    noise_bufs = self.get_noise_bufs(noise_bufs=noise_bufs, zero_noise=zero_noise)

    pts, rays_d, viewdirs, z_vals = nerf_utils.Render.prepare_nerf_inputs(
      focal=focals,
      img_size=img_size,
      cam_poses=cam_poses,
      near=near,
      far=far,
      **nerf_cfg)

    B, H, W, N, C = pts.shape
    pts = rearrange(pts, "b h w n c -> b (h w) n c")
    # normalized_pts = rearrange(normalized_pts, "b h w n c -> b (h w) n c")
    rays_d = rearrange(rays_d, "b h w c -> b (h w) c")
    viewdirs = rearrange(viewdirs, "b h w c -> b (h w) c")
    z_vals = rearrange(z_vals, "b h w n -> b (h w) n")

    # grad
    thumb_rgb, sdf, mask, xyz, features, eikonal_term = self.rays_forward(
      N_rays_forward=N_rays_forward,
      N_samples_forward=N_samples_forward,
      pts=pts,
      # normalized_pts=normalized_pts,
      rays_d=rays_d,
      viewdirs=viewdirs,
      z_vals=z_vals,
      near=near,
      far=far,
      style_render=style_render,
      eikonal_reg=eikonal_reg,
    )

    thumb_rgb = rearrange(thumb_rgb, "b (h w) c -> b c h w", h=H, w=W).contiguous()
    sdf = rearrange(sdf, "b (h w) n c -> b h w n c", h=H, w=W).contiguous()
    mask = rearrange(mask, "b (h w) c -> b c h w", h=H, w=W).contiguous()
    xyz = rearrange(xyz, "b (h w) c -> b c h w", h=H, w=W).contiguous()

    # decoder
    if enable_decoder:
      features = rearrange(features, "b (h w) c -> b c h w", h=H, w=W).contiguous()  # bug

      if renderer_detach:
        features = features.detach()

      rgb = self.decoder(features=features,
                         styles=style_decoder,
                         rgbd_in=None,
                         noise=noise_bufs).contiguous()  # bug
    else:
      # raise NotImplementedError
      rgb = thumb_rgb.clone()
      # rgb = rearrange(rgb, "b hw c -> b c hw 1").contiguous()  # bug
    # rgb = rearrange(rgb, "b c (h w) 1 -> b c h w", h=H, w=W).contiguous()

    ret_maps = {
      'rgb': rgb,
      'thumb_rgb': thumb_rgb,
      'style_decoder': style_decoder if path_reg else None,
      'eikonal_term': eikonal_term,
      'sdf': sdf if return_sdf else None,
      'xyz': xyz if return_xyz else None,
      'mask': mask[:, [0]],
      'depth': mask[:, [1]]
    }

    return ret_maps

  def create_noise_bufs(self, start_size, device):

    return self.decoder.create_noise_bufs(start_size=start_size, device=device)

  def _sample_sub_pixels(self,
                         fea_maps,
                         sample_idx_h,
                         sample_idx_w,
                         dim_h,
                         dim_w):

    samples_h = torch_utils.gather_points(points=fea_maps, sample_idx=sample_idx_h, dim=dim_h)

    samples_hw = torch_utils.gather_points(points=samples_h, sample_idx=sample_idx_w, dim=dim_w)

    return samples_hw

  def sample_sub_pixels(self,
                        pts,
                        rays_d,
                        viewdirs,
                        z_vals,
                        sample_idx_h,
                        sample_idx_w):
    """

    :param pts: (b h w n 3)
    :param rays_d: (b h w 3)
    :param viewdirs: (b h w 3)
    :param z_vals: (b h w n)
    :return:
    """
    pts = self._sample_sub_pixels(fea_maps=pts,
                                  sample_idx_h=sample_idx_h,
                                  sample_idx_w=sample_idx_w,
                                  dim_h=1,
                                  dim_w=2)
    rays_d = self._sample_sub_pixels(fea_maps=rays_d,
                                     sample_idx_h=sample_idx_h,
                                     sample_idx_w=sample_idx_w,
                                     dim_h=1,
                                     dim_w=2)
    viewdirs = self._sample_sub_pixels(fea_maps=viewdirs,
                                       sample_idx_h=sample_idx_h,
                                       sample_idx_w=sample_idx_w,
                                       dim_h=1,
                                       dim_w=2)
    z_vals = self._sample_sub_pixels(fea_maps=z_vals,
                                     sample_idx_h=sample_idx_h,
                                     sample_idx_w=sample_idx_w,
                                     dim_h=1,
                                     dim_w=2)

    return pts, rays_d, viewdirs, z_vals

  def part_grad_rays_forward(self,
                             N_rays_forward,
                             N_samples_forward,
                             idx_grad,
                             idx_no_grad,
                             pts,
                             # normalized_pts,
                             rays_d,
                             viewdirs,
                             z_vals,
                             near,
                             far,
                             style_render,
                             style_decoder,
                             noise_bufs,
                             eikonal_reg,
                             cam_poses,
                             project_noise,
                             mesh_path,
                             renderer_detach,
                             ):


    pts_grad = torch_utils.batch_gather_points(points=pts, idx_grad=idx_grad)
    # normalized_pts_grad = torch_utils.batch_gather_points(points=normalized_pts, idx_grad=idx_grad)
    rays_d_grad = torch_utils.batch_gather_points(points=rays_d, idx_grad=idx_grad)
    viewdirs_grad = torch_utils.batch_gather_points(points=viewdirs, idx_grad=idx_grad)
    z_vals_grad = torch_utils.batch_gather_points(points=z_vals, idx_grad=idx_grad)

    thumb_rgb_grad, sdf_grad, mask_grad, xyz_grad, rgb_grad, eikonal_term_grad = self.rays_forward(
      N_rays_forward=None,
      pts=pts_grad,
      # normalized_pts=normalized_pts_grad,
      rays_d=rays_d_grad,
      viewdirs=viewdirs_grad,
      z_vals=z_vals_grad,
      near=near,
      far=far,
      style_render=style_render,
      style_decoder=style_decoder,
      noise_bufs=noise_bufs,
      eikonal_reg=eikonal_reg,
      cam_poses=cam_poses,
      project_noise=project_noise,
      mesh_path=mesh_path,
      renderer_detach=renderer_detach)

    with torch.no_grad():
      pts_no_grad = torch_utils.batch_gather_points(points=pts, idx_grad=idx_no_grad)
      # normalized_pts_no_grad = torch_utils.batch_gather_points(points=normalized_pts, idx_grad=idx_no_grad)
      rays_d_no_grad = torch_utils.batch_gather_points(points=rays_d, idx_grad=idx_no_grad)
      viewdirs_no_grad = torch_utils.batch_gather_points(points=viewdirs, idx_grad=idx_no_grad)
      z_vals_no_grad = torch_utils.batch_gather_points(points=z_vals, idx_grad=idx_no_grad)

      thumb_rgb_no_grad, sdf_no_grad, mask_no_grad, xyz_no_grad, rgb_no_grad,\
      eikonal_term_no_grad = self.rays_forward(
        N_rays_forward=N_rays_forward,
        N_samples_forward=N_samples_forward,
        pts=pts_no_grad,
        # normalized_pts=normalized_pts_no_grad,
        rays_d=rays_d_no_grad,
        viewdirs=viewdirs_no_grad,
        z_vals=z_vals_no_grad,
        near=near,
        far=far,
        style_render=style_render,
        style_decoder=style_decoder,
        noise_bufs=noise_bufs,
        eikonal_reg=False,
        cam_poses=cam_poses,
        project_noise=project_noise,
        mesh_path=mesh_path,
        renderer_detach=renderer_detach)

    thumb_rgb = torch_utils.batch_scatter_points(idx_grad=idx_grad,
                                                 points_grad=thumb_rgb_grad,
                                                 idx_no_grad=idx_no_grad,
                                                 points_no_grad=thumb_rgb_no_grad,
                                                 dim=1)
    sdf = torch_utils.batch_scatter_points(idx_grad=idx_grad,
                                           points_grad=sdf_grad,
                                           idx_no_grad=idx_no_grad,
                                           points_no_grad=sdf_no_grad,
                                           dim=1)
    mask = torch_utils.batch_scatter_points(idx_grad=idx_grad,
                                            points_grad=mask_grad,
                                            idx_no_grad=idx_no_grad,
                                            points_no_grad=mask_no_grad,
                                            dim=1)
    xyz = torch_utils.batch_scatter_points(idx_grad=idx_grad,
                                           points_grad=xyz_grad,
                                           idx_no_grad=idx_no_grad,
                                           points_no_grad=xyz_no_grad,
                                           dim=1)
    rgb = torch_utils.batch_scatter_points(idx_grad=idx_grad,
                                           points_grad=rgb_grad,
                                           idx_no_grad=idx_no_grad,
                                           points_no_grad=rgb_no_grad,
                                           dim=2)

    return thumb_rgb, sdf, mask, xyz, rgb, eikonal_term_grad

  def rays_forward(self,
                   N_rays_forward,
                   pts, # (b hw n 3)
                   # normalized_pts,
                   rays_d,
                   viewdirs,
                   z_vals,
                   near,
                   far,
                   style_render,
                   eikonal_reg,
                   N_samples_forward=None,
                   ):

    B, N_rays, N_samples, _ = pts.shape

    if N_rays_forward is None:
      N_rays_forward = N_rays

    thumb_rgb_list = []
    features_list = []
    sdf_list = []
    mask_list = []
    xyz_list = []
    # eikonal_term_list = []
    rgb_list = []

    for idx in range(0, N_rays, N_rays_forward):

      nerf_inputs_dict = {}
      nerf_inputs_dict['pts'] = pts[:, idx: idx + N_rays_forward]
      # nerf_inputs_dict['normalized_pts'] = normalized_pts[:, idx: idx + N_rays_forward]
      nerf_inputs_dict['rays_d'] = rays_d[:, idx: idx + N_rays_forward]
      nerf_inputs_dict['viewdirs'] = viewdirs[:, idx: idx + N_rays_forward]
      nerf_inputs_dict['z_vals'] = z_vals[:, idx: idx + N_rays_forward]
      nerf_inputs_dict['near'] = near
      nerf_inputs_dict['far'] = far

      thumb_rgb, features, sdf, mask, xyz, eikonal_term = self.renderer(
        styles=style_render,
        return_eikonal=eikonal_reg,
        N_samples_forward=N_samples_forward,
        **nerf_inputs_dict)

      thumb_rgb_list.append(thumb_rgb)
      features_list.append(features)
      sdf_list.append(sdf)
      mask_list.append(mask)
      xyz_list.append(xyz)
      # eikonal_term_list.append(eikonal_term)

    thumb_rgb = torch.cat(thumb_rgb_list, dim=1)
    features = torch.cat(features_list, dim=1)
    # rgb = torch.cat(rgb_list, dim=-2)

    sdf = torch.cat(sdf_list, dim=1)

    mask = torch.cat(mask_list, dim=1)

    xyz = torch.cat(xyz_list, dim=1)

    return thumb_rgb, sdf, mask, xyz, features, eikonal_term

  @torch.no_grad()
  def get_mean_latent(self,
                      N_noises,
                      device):
    noises_renderer = torch.randn(N_noises, self.z_dim, device=device)
    style_renders = self.style(noises_renderer)
    style_render_mean = style_renders.mean(0, keepdim=True)

    noises_decoder = torch.randn(N_noises, self.z_dim, device=device)
    style_decoders = self.style_decoder(noises_decoder)
    style_decoder_mean = style_decoders.mean(0, keepdim=True)

    return style_render_mean, style_decoder_mean

  def mapping_networks(self,
                       zs,
                       truncation,
                       inject_index,
                       path_reg=False,
                       style_render=None,
                       style_decoder=None,
                       recompute_mean=False):

    if style_render is not None and style_decoder is not None:
      return style_render, style_decoder
    elif style_render is None and style_decoder is not None:
      raise NotImplementedError
    elif style_render is not None and style_decoder is None:
      raise NotImplementedError


    if truncation < 1:
      if recompute_mean or \
              not hasattr(self, 'style_render_mean') or \
              not hasattr(self, 'style_decoder_mean'):
        style_render_mean, style_decoder_mean = self.get_mean_latent(N_noises=10000, device=zs[0].device)
        self.style_render_mean = style_render_mean
        self.style_decoder_mean = style_decoder_mean
      else:
        style_render_mean = self.style_render_mean
        style_decoder_mean = self.style_decoder_mean
    else:
      style_render_mean = None
      style_decoder_mean = None

    style_render, latents = self.mapping_renderer(zs=[zs[0]],
                                                  truncation=truncation,
                                                  style_render_mean=style_render_mean)

    if path_reg:
      with torch.no_grad():
        style_decoder = self.mapping_decoder(latents=[zs[1]],
                                             truncation=truncation,
                                             style_decoder_mean=style_decoder_mean,
                                             inject_index=inject_index)

      style_decoder.requires_grad_(True)
    else:
      style_decoder = self.mapping_decoder(latents=[zs[1]],
                                           truncation=truncation,
                                           style_decoder_mean=style_decoder_mean,
                                           inject_index=inject_index)

    return style_render, style_decoder

  def mapping_decoder(self,
                      latents,
                      truncation,
                      style_decoder_mean,
                      inject_index=None):

    styles = [self.style_decoder(latent) for latent in latents]

    if truncation < 1:
      style_trunc = []
      for style in styles:
        style_trunc.append(style_decoder_mean + truncation * (style - style_decoder_mean))
      styles = style_trunc

    n_latent = self.decoder.n_latent

    if len(styles) < 2:
      style_decoder = styles[0].unsqueeze(1).repeat(1, n_latent, 1)

    else:
      if inject_index is None:
        # inject_index = random.randint(1, n_latent - 1)
        inject_index = n_latent
      style1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
      style2 = styles[1].unsqueeze(1).repeat(1, n_latent - inject_index, 1)

      style_decoder = torch.cat([style1, style2], 1)

    return style_decoder

  def create_mapping_decoder(self,
                             z_dim,
                             style_dim,
                             lr_mul_mapping,
                             N_layers):
    layers = [
      PixelNorm(),
      EqualLinear(z_dim, style_dim, lr_mul=lr_mul_mapping, activation="fused_lrelu"),
    ]
    self.module_name_list.extend(['style_decoder.0', 'style_decoder.1'])

    for i in range(N_layers - 1):
      layers.append(
        EqualLinear(style_dim, style_dim, lr_mul=lr_mul_mapping, activation="fused_lrelu")
      )
      self.module_name_list.append(f'style_decoder.{i + 2}')

    self.style_decoder = nn.Sequential(*layers)
    self.module_name_list.append('style_decoder')
    pass

  # def styles_and_noise_forward(self,
  def mapping_renderer(self,
                       zs,
                       truncation,
                       style_render_mean=None,
                       **kwargs):

    latents = [self.style(z) for z in zs]

    if truncation < 1:
      latents_trunc = []
      for latent in latents:
        latents_trunc.append(style_render_mean + truncation * (latent - style_render_mean))
      latents = latents_trunc

    style_render = latents[0].unsqueeze(1).repeat(1, self.N_layers_renderer + 1, 1)

    return style_render, latents

  def create_mapping_nerf(self,
                          z_dim,
                          style_dim,
                          N_layers):
    layers = []
    for i in range(N_layers):
      layers.append(
        MappingLinear(z_dim, style_dim, activation="fused_lrelu")
      )
      self.module_name_list.append(f"style.{i}")

    self.style = nn.Sequential(*layers)
    self.module_name_list.append('style')
    pass

  def make_noise(self):
    device = self.input.input.device

    noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

    for i in range(3, self.log_size + 1):
      for _ in range(2):
        noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

    return noises

  def get_latent(self, input):
    return self.style(input)

  def init_forward(self,
                   zs,
                   cam_poses,
                   focals,
                   img_size,
                   near,
                   far,
                   nerf_cfg):
    # latent = self.styles_and_noise_forward(styles)

    style_render, latents = self.mapping_renderer(zs=zs,
                                                  truncation=1)

    sdf, target_values = self.renderer.mlp_init_pass(cam_poses=cam_poses,
                                                     focals=focals,
                                                     img_size=img_size,
                                                     near=near,
                                                     far=far,
                                                     styles=style_render,
                                                     nerf_cfg=nerf_cfg)

    return sdf, target_values

  def get_ws(self,
             zs,
             truncation,
             device):

    w_render_mean, w_decoder_mean = self.get_mean_latent(N_noises=10000, device=device)

    w_render = w_render_mean + truncation * (self.style(zs[0]) - w_render_mean)
    w_decoder = w_decoder_mean + truncation * (self.style_decoder(zs[1]) - w_decoder_mean)

    w_render_plus = torch.zeros(w_render_mean.shape[0], self.N_layers_renderer + 1, w_render_mean.shape[-1],
                               device=device)
    w_render_plus.copy_(w_render[:, None, :])

    w_decoder_plus = torch.zeros(w_decoder_mean.shape[0], self.decoder.n_latent, w_decoder_mean.shape[-1],
                                device=device)
    w_decoder_plus.copy_(w_decoder[:, None, :])

    return w_render_plus, w_decoder_plus


############# Volume Renderer Building Blocks & Discriminator ##################
class VolumeRenderDiscConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
               padding=0, bias=True, activate=False):
    super(VolumeRenderDiscConv2d, self).__init__()

    self.conv = nn.Conv2d(in_channels, out_channels,
                          kernel_size, stride, padding, bias=bias and not activate)

    self.activate = activate
    if self.activate:
      self.activation = FusedLeakyReLU(out_channels, bias=bias, scale=1)
      bias_init_coef = np.sqrt(1 / (in_channels * kernel_size * kernel_size))
      nn.init.uniform_(self.activation.bias, a=-bias_init_coef, b=bias_init_coef)

  def forward(self, input):
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


class VolumeRenderDiscriminator(nn.Module):
  def __init__(self, opt):
    super().__init__()
    init_size = opt.renderer_spatial_output_dim
    self.viewpoint_loss = not opt.no_viewpoint_loss
    final_out_channel = 3 if self.viewpoint_loss else 1
    channels = {
      2: 400,
      4: 400,
      8: 400,
      16: 400,
      32: 256,
      64: 128,
      128: 64,
    }

    convs = [VolumeRenderDiscConv2d(3, channels[init_size], 1, activate=True)]

    log_size = int(math.log(init_size, 2))

    in_channel = channels[init_size]

    for i in range(log_size - 1, 0, -1):
      out_channel = channels[2 ** i]

      convs.append(VolumeRenderResBlock(in_channel, out_channel))

      in_channel = out_channel

    self.convs = nn.Sequential(*convs)

    self.final_conv = VolumeRenderDiscConv2d(in_channel, final_out_channel, 2)

  def forward(self, input):
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


######################### StyleGAN Discriminator ########################
class ResBlock(nn.Module):
  def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], merge=False):
    super().__init__()

    self.conv1 = ConvLayer(2 * in_channel if merge else in_channel, in_channel, 3)
    self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
    self.skip = ConvLayer(2 * in_channel if merge else in_channel, out_channel,
                          1, downsample=True, activate=False, bias=False)

  def forward(self, input):
    out = self.conv1(input)
    out = self.conv2(out)
    out = (out + self.skip(input)) / math.sqrt(2)

    return out


class Discriminator(nn.Module):
  def __init__(self, opt, blur_kernel=[1, 3, 3, 1]):
    super().__init__()
    init_size = opt.size

    channels = {
      4: 512,
      8: 512,
      16: 512,
      32: 512,
      64: 256 * opt.channel_multiplier,
      128: 128 * opt.channel_multiplier,
      256: 64 * opt.channel_multiplier,
      512: 32 * opt.channel_multiplier,
      1024: 16 * opt.channel_multiplier,
    }

    convs = [ConvLayer(3, channels[init_size], 1)]

    log_size = int(math.log(init_size, 2))

    in_channel = channels[init_size]

    for i in range(log_size, 2, -1):
      out_channel = channels[2 ** (i - 1)]

      convs.append(ResBlock(in_channel, out_channel, blur_kernel))

      in_channel = out_channel

    self.convs = nn.Sequential(*convs)

    self.stddev_group = 4
    self.stddev_feat = 1

    # minibatch discrimination
    in_channel += 1

    self.final_conv = ConvLayer(in_channel, channels[4], 3)
    self.final_linear = nn.Sequential(
      EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
      EqualLinear(channels[4], 1),
    )

  def forward(self, input):
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
