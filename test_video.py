import argparse
import os

import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import save_image
from function import coral 
import network as net
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
import time
from make_nnew_network import new_network
loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()

# read file function
def get_files(img_dir):
    files = os.listdir(img_dir)
    paths = []
    for x in files:
        paths.append(os.path.join(img_dir, x))
    return paths

# load image
def load_images(content_dir, style_dir):
    if os.path.isdir(content_dir):
        content_paths = get_files(content_dir)
    else:  # Single image file
        content_paths = [content_dir]
    if os.path.isdir(style_dir):
        style_paths = get_files(style_dir)
    else:  # Single image file
        style_paths = [style_dir]
    return content_paths, style_paths

# load model parameters
def load_weights(encoder, decoder, mcc):
    encoder.load_state_dict(torch.load(args.finetuned_encoder_path))
    mcc.load_state_dict(torch.load(args.finetuned_mcc_path))
    decoder.load_state_dict(torch.load(args.finetuned_decoder_path))
    print("encoder path:",args.finetuned_encoder_path)
    print("decoder path:",args.finetuned_decoder_path)
    print("mcc path:",args.finetuned_mcc_path)
    

# image pre-rocess
def test_transform(size, crop):
    transform_list = []
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

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

# style transfer fuction
def style_transfer(encoder, decoder, sa_module, content, style,alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    # startencoder = time.clock()
    style_fs, content_f, style_f=feat_extractor(encoder, content, style)
    # endencoder = time.clock()
    # print("The encoder run time is : %.03f seconds" %(endencoder-startencoder))
    # startmcc = time.clock()
    Fccc = sa_module(content_f,content_f)
    if interpolation_weights:
        _, C, H, W = Fccc.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = sa_module(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        Fccc=Fccc[0:1]
    else:
        feat = sa_module(content_f, style_f)
    
    feat = feat * alpha + Fccc * (1 - alpha)
    # endmcc = time.clock()
    # print("The mcc run time is : %.03f seconds" %(endmcc-startmcc))

    # startdecoder = time.clock()
    decoderresult = decoder(feat)
    # enddecoder = time.clock()
    # print("The decoder run time is : %.03f seconds" %(enddecoder-startdecoder))
    return decoderresult


def feat_extractor(encoder, content, style):

    style_fs = encoder(style)
    style_f = encoder(style)
    content_f = encoder(content)
    
    return style_fs,content_f, style_f

# image process, call style_tansfer
def image_process(content, style):
    content_tf1 = content_transform()
    content_frame = content_tf1(content) # content.convert("RGB")

    h, w, c = np.shape(content_frame)
    style_tf1 = style_transform(h, w)
    style = style_tf1(style.convert("RGB"))

    if yaml['preserve_color']:
        style = coral(style, content)

    style = style.to(device).unsqueeze(0)
    content = content_frame.to(device).unsqueeze(0)
    # print("------------",content.size(),style.size())
    with torch.no_grad():
        output = style_transfer(encoder, decoder, mcc, content, style,alpha)
    
    return output.squeeze(0).cpu()

# load video
def load_video(content_path,style_path, outfile):
    video = cv2.VideoCapture(content_path)

    rate = video.get(5)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  
    fps = int(rate)
    # print("=============================width,height============================",width,height)
    video_name = outfile + '/{:s}_stylized_{:s}{:s}'.format(
        splitext(basename(content_path))[0], splitext(basename(style_path))[0], '.mp4')

    videoWriter = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
                                  (int(width), int(height)))
    return video, videoWriter,int(width), int(height)
# save video 
def save_frame(output, videoWriter,width, height):
    output = output * 255
    output = output + 0.5   
    output = torch.tensor(torch.clamp(output, 0, 255).permute(1, 2, 0),dtype=torch.uint8).numpy()
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    # print("**********",output.shape)
    # print("right width and height size:",width,height)
    videoWriter.write(output) 

# video style transfer
def process_video(content_path, style_path, outfile):
    j = 0
    video, videoWriter,w,h= load_video(content_path, style_path, outfile)
    
    while (video.isOpened()):
        j = j + 1
        ret, frame = video.read()
        
        if not ret:
            break

        if j % 1 == False:
            # style transfer each frame
            style = Image.open(style_path)
            content = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            output = image_process(frame, style)
            # processing on the stylized results for storage in the video
            
            save_frame(output, videoWriter,w,h)

# image style transfer
def process_image(content_path, style_path, outfile):
    image_name = outfile + '/{:s}_stylized_{:s}{:s}'.format(
        splitext(basename(content_path))[0], splitext(basename(style_path))[0], '.jpg')
    # style transfer
    content = Image.open(content_path)
    style = Image.open(style_path)
    output = image_process(content, style)
    save_image(output, image_name)

def test(content_paths, style_paths):
    for content_path in content_paths:
        # process one content and one style
        outfile = output_path + '/' + splitext(basename(content_path))[0] + '/'
        if not os.path.exists(outfile):
            os.makedirs(outfile)

        # video style transfer
        if 'mp4' in content_path:
            for style_path in style_paths:
                process_video(content_path, style_path, outfile)
        # image style transfer
        else:
            for style_path in style_paths:
                process_image(content_path, style_path, outfile)

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', type=str,default="input/content", 
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str,default="input/style",
                        help='Directory path to a batch of style images')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save the output image(s)')
    parser.add_argument('--finetuned_encoder_path', type=str, 
                        default='./models/finetuned_encoder_iter_160000.pth')
    parser.add_argument('--finetuned_mcc_path', type=str, 
                        default='./models/finetuned_mcc_iter_160000.pth') 
    parser.add_argument('--finetuned_decoder_path', type=str, 
                        default='./models/finetuned_decoder_iter_160000.pth')
    
    parser.add_argument('--decoder_cfg', type=str, default='./models/pruned_decoder_iter_40000_cfg.txt') 
    parser.add_argument('--encoder_cfg', type=str, default='./models/newencoder_cfg.txt')
    
    parser.add_argument('--yaml_path', type=str, default='./test.yaml')
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--style_interpolation_weights', type=str, default="")
    
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    return args


if __name__ == '__main__':
    args = create_args()
    print("content_dir:",args.content_dir)
    print("style_dir:",args.style_dir)
    with open(args.yaml_path,'r') as file :
        yaml =yaml.load(file, Loader=yaml.FullLoader)
    alpha = args.a
    output_path = args.output_dir
    print("output path:",output_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.encoder_cfg, "r") as fp:
        cfgencoder = eval(fp.readline())
    encoder = new_network(cfg=cfgencoder).feature

    with open(args.decoder_cfg, "r") as fp:
        cfgdecoder = eval(fp.readline())
    decoder = new_network(cfg=cfgdecoder).feature
    mcc = net.MCC_Module(52)
    load_weights(encoder, decoder, mcc)
    print("encoder:",encoder)
    print("decoder:",decoder)
    print("mcc:",mcc)
    decoder.to(device)
    mcc.to(device)
    encoder.to(device)
    decoder.eval()
    mcc.eval()
    encoder.eval()

    content_paths, style_paths = load_images(args.content_dir, args.style_dir)
    # start = time.clock()
    test(content_paths, style_paths)
    # end = time.clock()
    # print("The function run time is : %.03f seconds" %(end-start))


