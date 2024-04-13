from os.path import basename
from os.path import splitext
import torch
from torchvision.utils import save_image
from thop.profile import profile
import argparse
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from make_nnew_network import new_network
loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()
import torch.nn as nn
import torch
from function import normal
import numpy as np

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
        # G_Fs = self.h(style_feat).view(-1,1,H*W) 
        G_Fs_sum = G_Fs_norm.view(B,C,H*W).sum(-1)
        FC_S = torch.bmm(G_Fs_norm,G_Fs_norm.permute(0,2,1)).view(B,C) /G_Fs_sum
        FC_S = self.fc(FC_S).view(B,C,1,1)
        out = F_Fc_norm*FC_S
        B,C,H,W = content_feat.size()
        out = out.contiguous().view(B,-1,H,W)
        out = self.out_conv(out)
        out = content_feat + out
        return out 
 
class MCC_Module(nn.Module):
    def __init__(self, in_dim):
        super(MCC_Module, self).__init__()    
        self.MCCN=MCCNet(in_dim)
    def forward(self, content_feats, style_feats):
        Fcsc = self.MCCN(content_feats, style_feats)
        return Fcsc

class Net(nn.Module):
    def __init__(self, encoder, mcc,decoder):
        super(Net, self).__init__()
        self.encoder = encoder
        self.mcc_module = mcc
        self.decoder = decoder

    def forward(self, content, style):
        style_feats = self.encoder(style)
        content_feats = self.encoder(content)
        Ics = self.decoder(self.mcc_module(content_feats, style_feats))
        return Ics

# style image pre-rocess
def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))

    transform_list = []
    transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

# content image pre-rocess
def content_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', type=str,default="input/content/bird.jpg")
    parser.add_argument('--style_dir', type=str,default="input/style/feathers.jpg")
    parser.add_argument('--encoder_path', type=str, 
                        default='./models/finetuned_encoder_iter_160000.pth')
    parser.add_argument('--mcc_path', type=str, 
                        default='./models/finetuned_mcc_iter_160000.pth') 
    parser.add_argument('--decoder_path', type=str, 
                        default='./models/finetuned_decoder_iter_160000.pth')
    parser.add_argument('--decoder_cfg', type=str, 
                        default='./models/pruned_decoder_iter_40000_cfg.txt') 
    parser.add_argument('--encoder_cfg', type=str, 
                        default='./models/newencoder_cfg.txt')
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--style_interpolation_weights', type=str, default="")
    
    args = parser.parse_args()
    return args

def load_models():
    with open(args.encoder_cfg, "r") as fp:
        cfgencoder = eval(fp.readline())
    encoder = new_network(cfg=cfgencoder).feature
    encoder.load_state_dict(torch.load(args.encoder_path))
    print("encoder path:",args.encoder_path)

    with open(args.decoder_cfg, "r") as fp:
        cfgdecoder = eval(fp.readline())
    decoder = new_network(cfg=cfgdecoder).feature
    decoder.load_state_dict(torch.load(args.decoder_path))
    print("decoder path:",args.decoder_path)
    mcc = MCC_Module(52)
    mcc.load_state_dict(torch.load(args.mcc_path))
    print("mcc path:",args.mcc_path)
    return encoder, mcc, decoder


if __name__ == '__main__':
    args = create_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    encoder, mcc, decoder = load_models()

    model = Net(encoder,mcc,decoder)
    model.eval()
    model.to(device)
    print(model)
    content = Image.open(args.content_dir)
    style = Image.open(args.style_dir)
    content_tf1 = content_transform()
    content_frame = content_tf1(content)

    h, w, c = np.shape(content_frame)
    style_tf1 = style_transform(h, w)
    style = style_tf1(style.convert("RGB"))
    style = style.to(device).unsqueeze(0)
    content = content_frame.to(device).unsqueeze(0)
    print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
    print("---|---|---")
    ops, params = profile(model, (content,style), verbose=False)
    print(
        "model | %.2f | %.2f" % (params / (1000 ** 2), ops / (1000 ** 3))
    )

    cf = model.encoder(content)
    sf = model.encoder(style)
    ops, params = profile(model.encoder, (content,), verbose=False)
    print(
        "encoder_content | %.2f | %.2f" % (params / (1000 ** 2), ops / (1000 ** 3))
    )
    ops, params = profile(model.encoder, (style,), verbose=False)
    print(
        "encoder_style | %.2f | %.2f" % (params / (1000 ** 2), ops / (1000 ** 3))
    )
    mcc_f = model.mcc_module(cf,sf)
    ops, params = profile(model.mcc_module, (cf,sf), verbose=False)
    print(
        "mcc | %.2f | %.2f" % (params / (1000 ** 2), ops / (1000 ** 3))
    )
    output = model.decoder(mcc_f)
    ops, params = profile(model.decoder, (mcc_f,), verbose=False)
    print(
        "decoder | %.2f | %.2f" % (params / (1000 ** 2), ops / (1000 ** 3))
    )



