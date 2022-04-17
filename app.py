import os
import numpy as np
import time
import random
import torch
import torchvision.transforms as transforms
#import requests
import gradio as gr
import matplotlib.pyplot as plt

from models import get_model
from dotmap import DotMap
from PIL import Image


# args
args = DotMap()
args.deploy = 'vanilla'
args.arch = 'dino_small_patch16'
args.device = 'cuda:7'
args.resume = '/fast_scratch/hushell/fluidstack/FS125_few-shot-transformer/outputs/dinosmall_1e-4/best_converted.pth'
args.api_key = 'AIzaSyAFkOGnXhy-2ZB0imDvNNqf2rHb98vR_qY'
args.cx = '06d75168141bc47f1'


# model
device = torch.device(args.device)
model = get_model(args)
model.to(device)
checkpoint = torch.load(args.resume, map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=True)


# image transforms
def test_transform():
    def _convert_image_to_rgb(im):
        return im.convert('RGB')

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ])

preprocess = test_transform()

@torch.no_grad()
def denormalize(x, mean, std):
    # 3, H, W
    t = x.clone()
    t.mul_(std).add_(mean)
    return torch.clamp(t, 0, 1)


# Google image search
from google_images_search import GoogleImagesSearch

# define search params
# option for commonly used search param are shown below for easy reference.
# For param marked with '##':
#   - Multiselect is currently not feasible. Choose ONE option only
#   - This param can also be omitted from _search_params if you do not wish to define any value
_search_params = {
    'q': '...',
    'num': 10,
    'fileType': 'png', #'jpg|gif|png',
    'rights': 'cc_publicdomain', #'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived',
    #'safe': 'active|high|medium|off|safeUndefined', ##
    'imgType': 'photo', #'clipart|face|lineart|stock|photo|animated|imgTypeUndefined', ##
    #'imgSize': 'huge|icon|large|medium|small|xlarge|xxlarge|imgSizeUndefined', ##
    #'imgDominantColor': 'black|blue|brown|gray|green|orange|pink|purple|red|teal|white|yellow|imgDominantColorUndefined', ##
    'imgColorType': 'color', #'color|gray|mono|trans|imgColorTypeUndefined' ##
}


# Gradio UI
def inference(query, labels, n_supp=10):
    '''
    query: PIL image
    labels: list of class names
    '''
    labels = labels.split(',')

    with torch.no_grad():
        # query image
        query = preprocess(query).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 3, H, W)

        supp_x = []
        supp_y = []
        supp_grid = []

        # search support images
        for idx, y in enumerate(labels):
            with GoogleImagesSearch(args.api_key, args.cx) as gis:
                _search_params['q'] = y
                _search_params['num'] = n_supp
                gis.search(search_params=_search_params)

                for j, x in enumerate(gis.results()):
                    #url = x.url
                    #x_im = Image.open(requests.get(url, stream=True).raw)
                    x.download('./')
                    x_im = Image.open(x.path)
                    x_im = preprocess(x_im) # (3, H, W)
                    supp_x.append(x_im)
                    supp_y.append(idx)
                    if j == 0:
                        supp_grid.append(denormalize(x_im))

        print('Searching for support images is done.')

        supp_x = torch.stack(supp_x, dim=0).unsqueeze(0).to(device) # (1, n_supp*n_labels, 3, H, W)
        supp_y = torch.tensor(supp_y).long().unsqueeze(0).to(device) # (1, n_supp*n_labels)

        with torch.cuda.amp.autocast(True):
            output = model(supp_x, supp_y, query) # (1, 1, n_labels)

        probs = output.softmax(dim=-1).detach().cpu().numpy()

        supp_grid = torch.stack(

        return {k: float(v) for k, v in zip(labels, probs[0, 0])}


# DEBUG
#query = Image.open('../labrador-puppy.jpg')
#labels = 'dog, cat'
#output = inference(query, labels, n_supp=1)
#print(output)


gr.Interface(fn=inference,
             inputs=[
                 gr.inputs.Image(label="Image to classify:", type="pil"),
                 gr.inputs.Textbox(lines=1, label="Comma separated class hypotheses:", placeholder="Enter class names separated by ','",),
                 gr.inputs.Number(default=1, label="Number of support examples from Google")
             ],
             theme="grass",
             outputs=[
                 gr.outputs.Label(),
                 gr.outputs.Image(type="auto", label="Support examples"),
             ],
             description="PMF few-shot learning with Google image search").launch()
