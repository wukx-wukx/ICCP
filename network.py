import torch.nn as nn
import torch
from function import normal

class MCCNet(nn.Module):
    def __init__(self, in_dim):
        super(MCCNet, self).__init__()
        self.f = nn.Conv2d(in_dim , int(in_dim ), (1,1))
        self.g = nn.Conv2d(in_dim , int(in_dim ) , (1,1))
        self.h = nn.Conv2d(in_dim  ,int(in_dim ) , (1,1))
        self.out_conv = nn.Conv2d(int(in_dim ), in_dim, (1, 1))
        self.fc = nn.Linear(in_dim, in_dim)

    def forward(self,content_feat,style_feat):
        B,C,H,W = content_feat.size()

        F_Fc_norm  = self.f(normal(content_feat))

        B,C,H,W = style_feat.size()
        G_Fs_norm =  self.g(normal(style_feat)).view(-1,1,H*W) 
        G_Fs = self.h(style_feat).view(-1,1,H*W) 
        G_Fs_sum = G_Fs_norm.view(B,C,H*W).sum(-1)
        FC_S = torch.bmm(G_Fs_norm,G_Fs_norm.permute(0,2,1)).view(B,C) /G_Fs_sum  #14
        FC_S = self.fc(FC_S).view(B,C,1,1)

        
        out = F_Fc_norm*FC_S
        
        B,C,H,W = content_feat.size()
       
        out = out.contiguous().view(B,-1,H,W)
        out = self.out_conv(out)

        out = content_feat + out
        return out # , FC_S
 

class MCC_Module(nn.Module):
    def __init__(self, in_dim):
        super(MCC_Module, self).__init__() 
        self.MCCN=MCCNet(in_dim)
    def forward(self, content_feats, style_feats):
        Fcsc = self.MCCN(content_feats, style_feats)
        return Fcsc


