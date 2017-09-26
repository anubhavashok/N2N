from utils import *
import torch


class Layer():
    def __init__(self, layer):
        self._layer = layer
        self.type = getLayerType(layer)
        k = getattr(layer, 'kernel_size', 0)
        k = k[0] if type(k) is tuple else k
        s = getattr(layer, 'stride', 0)
        s = s[0] if type(s) is tuple else s
        o = getattr(layer, 'out_channels', 0)
        o = getattr(layer, 'out_features', o)
        p = getattr(layer, 'padding', 0)
        p = p[0] if type(p) is tuple else p
        skipstart = layer.skipstart if hasattr(layer, 'skipstart') else 0
        skipend = layer.skipend if hasattr(layer, 'skipend') else 0
        self.k = k
        self.s = s
        self.o = o
        self.p = p
        self.skipstart = skipstart
        self.skipend = skipend
                
    def getRepresentation(self, skipSupport=False):
        rep = [self.type, self.k, self.s, self.o, self.p]
        if skipSupport:
            rep.extend([self.skipstart, self.skipend])
        return rep

    def toTorchTensor(self, skipSupport=False):
        t = torch.Tensor(self.getRepresentation(skipSupport))
        t = t.unsqueeze(0)
        t = t.unsqueeze(0)
        return t
