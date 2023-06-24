import json
import os.path
from typing import Union, List, Dict, Any, cast

import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import ClassifierHead
from timm.models.registry import register_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tv_trans

from tl2.proj.fvcore import MODEL_REGISTRY, global_cfg
from tl2.proj.fvcore.checkpoint import Checkpointer
from tl2.proj.pytorch.pytorch_hook import FeatureExtractor
from tl2.modelarts import moxing_utils


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (1, 1),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'features.0', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = {
    'vgg16': _cfg(url='https://download.pytorch.org/models/vgg16-397923af.pth'),
    'vgg19': _cfg(url='https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'),
}


cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class ConvMlp(nn.Module):

    def __init__(self, in_features=512, out_features=4096, kernel_size=7, mlp_ratio=1.0,
                 drop_rate: float = 0.2, act_layer: nn.Module = None, conv_layer: nn.Module = None):
        super(ConvMlp, self).__init__()
        self.input_kernel_size = kernel_size
        mid_features = int(out_features * mlp_ratio)
        self.fc1 = conv_layer(in_features, mid_features, kernel_size, bias=True)
        self.act1 = act_layer(False)
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = conv_layer(mid_features, out_features, 1, bias=True)
        self.act2 = act_layer(False)

    def forward(self, x):
        if x.shape[-2] < self.input_kernel_size or x.shape[-1] < self.input_kernel_size:
            # keep the input size >= 7x7
            output_size = (max(self.input_kernel_size, x.shape[-2]), max(self.input_kernel_size, x.shape[-1]))
            x = F.adaptive_avg_pool2d(x, output_size)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


class VGG(nn.Module):

    def __init__(
        self,
        cfg: List[Any],
        num_classes: int = 1000,
        in_chans: int = 3,
        output_stride: int = 32,
        mlp_ratio: float = 1.0,
        act_layer: nn.Module = nn.ReLU,
        conv_layer: nn.Module = nn.Conv2d,
        norm_layer: nn.Module = None,
        global_pool: str = 'avg',
        drop_rate: float = 0.,
    ) -> None:
        super(VGG, self).__init__()
        assert output_stride == 32
        self.num_classes = num_classes
        self.num_features = 4096
        self.drop_rate = drop_rate
        self.feature_info = []
        prev_chs = in_chans
        net_stride = 1
        pool_layer = nn.MaxPool2d
        layers: List[nn.Module] = []
        for v in cfg:
            last_idx = len(layers) - 1
            if v == 'M':
                self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=f'features.{last_idx}'))
                layers += [pool_layer(kernel_size=2, stride=2)]
                net_stride *= 2
            else:
                v = cast(int, v)
                conv2d = conv_layer(prev_chs, v, kernel_size=3, padding=1)
                if norm_layer is not None:
                    layers += [conv2d, norm_layer(v), act_layer(inplace=False)]
                else:
                    layers += [conv2d, act_layer(inplace=False)]
                prev_chs = v
        self.features = nn.Sequential(*layers)
        self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=f'features.{len(layers) - 1}'))
        self.pre_logits = ConvMlp(
            prev_chs, self.num_features, 7, mlp_ratio=mlp_ratio,
            drop_rate=drop_rate, act_layer=act_layer, conv_layer=conv_layer)
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

        self._initialize_weights()

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.head = ClassifierHead(
            self.num_features, self.num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _filter_fn(state_dict):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        k_r = k
        k_r = k_r.replace('classifier.0', 'pre_logits.fc1')
        k_r = k_r.replace('classifier.3', 'pre_logits.fc2')
        k_r = k_r.replace('classifier.6', 'head.fc')
        if 'classifier.0.weight' in k:
            v = v.reshape(-1, 512, 7, 7)
        if 'classifier.3.weight' in k:
            v = v.reshape(-1, 4096, 1, 1)
        out_dict[k_r] = v
    return out_dict


def _create_vgg(variant: str, pretrained: bool, **kwargs: Any) -> VGG:
    cfg = variant.split('_')[0]
    # NOTE: VGG is one of the only models with stride==1 features, so indices are offset from other models
    out_indices = kwargs.get('out_indices', (0, 1, 2, 3, 4, 5))
    model = build_model_with_cfg(
        VGG, variant, pretrained,
        default_cfg=default_cfgs[variant],
        model_cfg=cfgs[cfg],
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        pretrained_filter_fn=_filter_fn,
        **kwargs)
    return model


@register_model
def vgg16_conv(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(**kwargs)
    return _create_vgg('vgg16', pretrained=pretrained, **model_args)


@register_model
def vgg19_conv(pretrained: bool = False, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    """
    model_args = dict(**kwargs)
    return _create_vgg('vgg19', pretrained=pretrained, **model_args)




@MODEL_REGISTRY.register(name_prefix=__name__)
class VGG16ConvLoss(torch.nn.Module):
  def __init__(self,
               model_name='vgg16_conv',
               downsample_size=-1,
               use_stat_loss=False,
               layers=None,
               loss_w_dict=None,
               rank=0,
               **kwargs):
    super().__init__()

    self.downsample_size = downsample_size
    self.use_stat_loss = use_stat_loss

    assert model_name in ['vgg16_relu',
                          'vgg16_conv',
                          'vgg16_conv_random']

    self.mean = IMAGENET_DEFAULT_MEAN
    self.std = IMAGENET_DEFAULT_STD
    self.transform = tv_trans.Normalize(mean=self.mean, std=self.std)

    if layers is None:
      layers = self.layers
    print(f"{model_name} layers: {layers}")

    if model_name in ['vgg16_relu']:
      net = timm.create_model('vgg16', pretrained=True, features_only=True)
    elif model_name in ['vgg16_conv']:
      moxing_utils.copy_data(rank=rank, global_cfg=global_cfg,
                             datapath_obs=f"keras/cache/torch/hub/checkpoints/vgg16-397923af.pth",
                             datapath=os.path.expanduser("~/.cache/torch/hub/checkpoints/vgg16-397923af.pth"))
      net = timm.create_model('vgg16_conv', pretrained=True, features_only=True)
    elif model_name in ['vgg16_conv_random']:
      net = timm.create_model('vgg16_conv', pretrained=False, features_only=True)
      print("random initialize vgg16_conv ...")

    self.net = FeatureExtractor(net, layers=layers)

    self.loss_w_dict = loss_w_dict
    if loss_w_dict is None:
      self.loss_w_dict = self.loss_weight(name='vgg16_conv_1024')
    print(f"{model_name} loss_w_dict: {json.dumps(self.loss_w_dict)}")
    pass

  @property
  def layers(self):
    layers = ['features_2',
              'features_7',
              'features_14',
              'features_21',
              'features_28']
    return layers

  def loss_weight(self, name):
    if name == 'vgg16_conv_1024':
      # VGG16 conv perceptual loss for 1024x1024.
      loss_w_dict = {
        'features_2': 0.0002,
        'features_7': 0.0001,
        'features_14': 0.0001,
        'features_21': 0.0002,
        'features_28': 0.0005,
      }
    elif name == 'vgg16_conv_256':
      loss_w_dict = {
        'features_2': 0.001,
        'features_7': 0.0006,
        'features_14': 0.0005,
        'features_21': 0.0005,
        'features_28': 0.001,
      }
    elif name == 'vgg16_relu_1024':
      # VGG16 conv perceptual loss for 1024x1024.
      loss_w_dict = {
        'features_2': 0.0006,
        'features_7': 0.0004,
        'features_14': 0.0004,
        'features_21': 0.0007,
        'features_28': 0.007,
      }
    elif name == 'vgg16_relu_256':
      # VGG16 conv perceptual loss for 1024x1024.
      loss_w_dict = {
        'features_2': 0.001,
        'features_7': 0.001,
        'features_14': 0.001,
        'features_21': 0.002,
        'features_28': 0.01,
      }
    else:
      assert 0
    return loss_w_dict


  def forward(self,
              x,
              *args,
              loss_w_dict=None,
              use_stat_loss=None,
              **kwargs):
    """
    x: [-1 , 1]
    """
    self.net.eval()

    if use_stat_loss is None:
      use_stat_loss = self.use_stat_loss

    x = (x + 1) / 2.
    x = self.transform(x)
    if self.downsample_size > 0:
      downsample_size = (self.downsample_size, self.downsample_size)
      x = F.interpolate(x, size=downsample_size, mode='area')

    feas_dict = self.net(x)
    feas = []
    for k, v in feas_dict.items():
      fea = v
      # b, c, h, w = fea.shape
      if use_stat_loss:
        fea = self.stat_loss(fea)
      else:
        fea = fea.flatten(start_dim=1)

      if loss_w_dict is None:
        fea = fea * self.loss_w_dict[k]
      else:
        fea = fea * loss_w_dict[k]
      feas.append(fea)
    feas = torch.cat(feas, dim=1)
    return feas

  def stat_loss(self, fea):
    fea = fea.flatten(start_dim=2)
    var_mean = torch.var_mean(fea, dim=2)
    fea = torch.cat(var_mean, dim=1)
    return fea

