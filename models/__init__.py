from .segmentors import DeSPARNetwork
from .backbones import PVTv2_DGL_Encoder
from .decode_heads import UniversalDecoder

__all__ = ['DeSPARNetwork', 'PVTv2_DGL_Encoder', 'UniversalDecoder']