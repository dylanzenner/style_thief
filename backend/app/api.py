from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from typing import List
from starlette.responses import StreamingResponse
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.model = models.vgg19(pretrained=True)
    
    def forward(self, img):
        features = []
        for layer in self.model.features:
            img = layer(img)
            if type(layer) == nn.Conv2d:
                features.append(img)
        return features


app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def receive_file(base_image: bytes = File(), style_image: bytes = File()):
    start_time = time.time()
    # load the images
    base_img = Image.open(io.BytesIO(base_image))
    style_img = Image.open(io.BytesIO(style_image))

    # resize the images to 224 x 224 and normalize
    resize_image = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # unsqueeze the images to add another dimension for batch size and use the CPU
    base_img_resized = resize_image(base_img).unsqueeze(0).to('cpu')
    style_img_resized = resize_image(style_img).unsqueeze(0).to('cpu')

    target = base_img_resized.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([target], lr=0.03)

    vgg = VGG19().to('cpu').eval()

    # 9126.44554066658 seconds for:
    # learning rate: 0.001
    # 5000 epochs

    for epoch in range(100):
        print('Epoch: {}'.format(epoch))
        
        target_features = vgg(target)
        base_features = vgg(base_img_resized)
        style_features = vgg(style_img_resized)

        style_loss = 0
        base_loss = 0

        for tf, bf, sf in zip(target_features, base_features, style_features):
            base_loss += torch.mean((tf - bf) ** 2)

            _, c, h, w = tf.size()
            tf = tf.view(c, h * w)
            sf = sf.view(c, h * w)

            tf = torch.mm(tf, tf.t())
            sf = torch.mm(sf, sf.t())

            style_loss += torch.mean((tf - sf) ** 2 / (c * h * w))

        total_loss = 1e3 * base_loss + 1e6 * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        
    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    target_img = denorm(target.squeeze()).clamp_(0, 1)
    final = transforms.ToPILImage()(target_img)

    bytes_image = io.BytesIO()
    final.save(bytes_image, 'JPEG')
    bytes_image.seek(0)
    print(time.time() - start_time)


    return StreamingResponse(content=bytes_image, media_type="image/jpeg")

