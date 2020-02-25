import torch
from torch import nn


class Res2Conv(nn.Module):
    def __init__(self, features, stride=1, scale_num=1, cardinality=1):
        super(Res2Conv, self).__init__()
        assert scale_num >= 2 # it will be standard conv when scale_num=1
        assert features % scale_num == 0
        self.invi_features = int(features / scale_num)
        self.convs = nn.ModuleList()
        for i in range(scale_num - 1):
            self.convs.append(
                nn.Conv2d(self.invi_features, self.invi_features, 3, stride=stride, padding=1, groups=cardinality)
            )

    def forward(self, x):
        feas = x[:,:self.invi_feattures,:,:]
        fea = 0
        for i, conv in enumerate(self.convs):
            fea += x[:,self.invi_feattures*(i+1):self.invi_feattures*(i+2),:,:]
            fea = conv(fea)
            feas = torch.cat([feas, fea], dim=1)
        return feas


if __name__=="__main__":
    res2conv = Res2Conv(64, 1, scale_num=4, cardinality=1)
    res2conv.cuda()
    x = torch.rand([8, 64, 28, 28]).cuda()
    y = res2conv(x)
    print(x.shape)
    print(y.shape)
