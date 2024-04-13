import torch.nn as nn
# create a new structure
class new_network(nn.Module):
    def __init__(self,cfg):
        super(new_network, self).__init__()
        self.feature = self.make_layers(cfg)

    def make_layers(self, cfg):
        layers = []
        if 'M' in cfg: # encoder
            in_channels = 3
        else:          # decoder
            in_channels = 52
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)]
            elif v == 'U':
                layers += [nn.Upsample(scale_factor=2, mode='nearest')]
            elif v == 'RP':
                layers += [nn.ReflectionPad2d((1, 1, 1, 1))]
            elif v == 'Re': 
                layers += [nn.ReLU()]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3,  bias=False)
                layers += [conv2d, nn.BatchNorm2d(v)] 
                in_channels = v
        k = 0
        if 'M' in cfg: 
            for i in range(len(layers)):
                if isinstance(layers[i], nn.Conv2d):
                    k = i
                    break
            out_channel = layers[k].out_channels
            layers.remove(layers[k])
            layers.insert(k,nn.Conv2d(3, out_channel, kernel_size=1,  bias=False))
        return nn.Sequential(*layers[:-1]) # remove the last BN layer of encoder and decoder



