from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models



def resize_image(img, image_size = 224):

    loader = transforms.Compose([
        transforms.Resize((image_size, image_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    resized_img = loader(img).unsqueeze(0).to('cpu')
    return resized_img

def content_loss(model, content_img, content_layer=4, epochs=1000, lr=1e-1):
    content_features = model(content_img)
    img = 1e-1*torch.randn_like(content_img)
    img.requires_grad_(True)

    opt = torch.optim.Adam([img], lr = lr)

    for epoch in range(epochs):
        print('epoch: {}'.format(epoch))
        opt.zero_grad()
        features = model(img)
        content = features[content_layer]
        content_target = content_features[content_layer].detatch()
        loss = ((content - content_target) ** 2).sum()
        loss.backward()
        opt.step()
    return img












def training_loop(model, base_img, base_img_layer=4, epochs=1000, lr=1e-1):

    base_features = model(base_img)
    noise_img = torch.randn(base_img.data.shape, device='cpu', requires_grad=True)
    optimizer = optim.Adam([noise_img], lr = lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        features = model(noise_img)
        content = features[base_img_layer]
        content_target = base_features[base_img_layer].detatch()
        loss = ((content - content_target) ** 2).sum()
        loss.backward()
        optimizer.step()
    return resize_images(noise_img)



class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.model = models.vgg19(pretrained=True).eval().to('cpu')


    def forward(self, x):

        features = []

        for layer in self.model.features:
            x = layer(x)
            if type(x) == nn.Conv2d:
                features.append(x)
        return features




    


    

