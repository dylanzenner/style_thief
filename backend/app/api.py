from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from typing import List
from starlette.responses import StreamingResponse
import cv2
import numpy as np
from style import VGG19, content_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models

import copy



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
    # load the images
    base_img = Image.open(io.BytesIO(base_image))
    style_img = Image.open(io.BytesIO(style_image))

    # resize the images to 224 x 224
    resize_image = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    base_img_resized = resize_image(base_img).unsqueeze(0).to('cpu')
    style_img_resized = resize_image(style_img).unsqueeze(0).to('cpu')
    
    # generated a noise image
    noise_img = torch.randn(base_img_resized.data.shape, device='cpu', requires_grad=True)
    
    # instantiate the model
    model = VGG19()

    print(content_loss(model, base_img_resized ,content_layer=0))





    # total_steps = 100
    # learning_rate = 0.001
    # alpha = 1
    # beta = 0.01
    # optimizer = optim.Adam([noise_img], lr = learning_rate)

    # for step in range(total_steps):
    #     noise_features = model(noise_img)
    #     base_features = model(base_img_resized)
    #     style_features = model(style_img_resized)

    #     style_loss = 0
    #     base_loss = 0
    #     print('Starting: {}'.format(step))
    #     for noise_feature, base_feature, style_feature in zip(noise_features, base_features, style_features):
    #         _, c, h, w = noise_feature.shape
            
    #         base_loss += torch.mean((noise_feature - base_feature) ** 2)

    #         G = noise_feature.view(c, h * w).mm(noise_feature.view(c, h * w).t())
    #         A = style_feature.view(c, h * w).mm(style_feature.view(c, h * w).t())

        #     style_loss += torch.mean((G - A) ** 2)

        # total_loss = alpha * base_loss + beta * style_loss
        # optimizer.zero_grad()
        # total_loss.backward()
        # optimizer.step()
        # if step % 20 == 0:
        #     denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
        #     stylized_img = noise_img.clone().squeeze()
        #     stylized_img = denorm(stylized_img).clamp_(0, 1)
        #     final = transforms.ToPILImage()(stylized_img) # make sure tensor is on cpu
        #     final.show()


        #     bytes_image = io.BytesIO()
        #     final.save(bytes_image, 'JPEG')
        #     bytes_image.seek(0)
        #     print(total_loss)
        #     print(stylized_img.shape)
        #     print('YESSS')

    # final = transforms.ToPILImage()(stylized_img) # make sure tensor is on cpu
    # final.show()


    # bytes_image = io.BytesIO()
    # final.save(bytes_image, 'JPEG')
    # bytes_image.seek(0)


    # return StreamingResponse(content=bytes_image, media_type="image/jpeg")

