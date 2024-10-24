import os 
import argparse
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
from model.vgg import VGG


img_size = 512 
prep = transforms.Compose([transforms.Resize(img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
postpb = transforms.Compose([transforms.ToPILImage()])


# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)

def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img

def load_image(img_path):
    img = Image.open(img_path)
    img_torch = prep(img)
    if torch.cuda.is_available():
        img_torch = Variable(img_torch.unsqueeze(0).cuda())
    else:
        img_torch = Variable(img.unsqueeze(0))
    return img_torch 

def simple_split_list(data_list, n_parts):
    # Using Python's built-in functionality to split the list into `n_parts`
    return [data_list[i::n_parts] for i in range(n_parts)]

def main(data_path, dst_path, model_dir, idx): 
    # These are good weights settings:
    style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
    content_weights = [1e0]
    weights = style_weights + content_weights

    # Define layers, loss functions, weights and compute optimization targets
    style_layers = ['r11','r21','r31','r41', 'r51'] 
    content_layers = ['r42']
    loss_layers = style_layers + content_layers
    loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    if torch.cuda.is_available():
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

    #run style transfer
    max_iter = 500
    show_iter = 50

    # Get network
    vgg = VGG()
    vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
    for param in vgg.parameters():
        param.requires_grad = False
    if torch.cuda.is_available():
        vgg.cuda()

    data_path_list = glob(os.path.join(data_path, '*/*.jpg'))

    # Split the `data_path_list` into 12 parts
    split_data_path_lists_simple = simple_split_list(data_path_list, 24)
    global_progress = tqdm(split_data_path_lists_simple[idx], desc=f'Stylization {idx} index list images')

    for shape_path in global_progress:
        shape_category = shape_path.split('/')[-2]
        os.makedirs(os.path.join(dst_path, shape_category), exist_ok=True)
        shape_file_name = os.path.splitext(shape_path.split('/')[-1])[0]
        
        # Progress bar for style-content pairs
        local_progress = tqdm(data_path_list, desc=f"Stylization {shape_file_name}_{shape_category}", leave=False)
        for texture_path in local_progress:
            texture_category = texture_path.split('/')[-2]
            os.makedirs(os.path.join(dst_path, shape_category, texture_category), exist_ok=True)
            texture_file_name = os.path.splitext(texture_path.split('/')[-1])[0]
            local_progress.set_description(f"Stylization {shape_file_name}_{shape_category} -> {texture_file_name}_{texture_category}")
            
            # ignore the same class between shape and texture
            if texture_category == shape_category:
                continue
            
            content_image = load_image(shape_path)

            # load texture image(style image)
            style_image = load_image(texture_path)
            opt_image = Variable(content_image.data.clone(), requires_grad=True)

            #compute optimization targets
            style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
            content_targets = [A.detach() for A in vgg(content_image, content_layers)]
            targets = style_targets + content_targets

            optimizer = optim.LBFGS([opt_image])
            n_iter=[0]

            while n_iter[0] <= max_iter:

                def closure():
                    optimizer.zero_grad()
                    out = vgg(opt_image, loss_layers)
                    layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
                    loss = sum(layer_losses)
                    loss.backward()
                    n_iter[0]+=1
                    #print loss
                    if n_iter[0]%show_iter == (show_iter-1):
                        local_progress.write(f'Iteration: {n_iter[0]+1}, loss: {loss.item():.6f}')
                    return loss
                
                optimizer.step(closure)

            #display result
            out_img = postp(opt_image.data[0].cpu().squeeze())
            out_img.save(os.path.join(dst_path, shape_category, f"{shape_file_name}_{texture_file_name}.jpg"))
        
    return        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=1)
    args = parser.parse_args()

    data_path = f"/workspace/seungah.lee/data/pascal_voc/9_class/image/trainval/"
    dst_path = f"/nas2/lait/1000_Members/seungah.lee/ssl-shape-texture/neural-style-transfer/voc/9_class/trainval"
    model_dir = f"/workspace/seungah.lee/style-transfer/neural-style-transfer/checkpoint/"
    main(data_path, dst_path, model_dir, args.idx)

        

