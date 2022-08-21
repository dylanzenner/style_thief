from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from typing import List
from starlette.responses import StreamingResponse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import asyncio
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

async def concurrent_target_features(c, w, h, tf):
    tf = tf.view(c, h * w)
    tf = torch.mm(tf, tf.t())
    return tf

async def concurrent_style_features(c, w, h, sf):
    sf = sf.view(c, h * w)
    sf = torch.mm(sf, sf.t())
    return sf

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
    alpha = 1e3
    beta = 1e6
    learning_rate = 0.03
    optimizer = torch.optim.Adam([target], lr=learning_rate)

    vgg = VGG19().to('cpu').eval()
    
    # try and optimize for only 100 epochs in order to have a fast web application
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

            # async concurrent speeds up the algorithm by roughly a minute
            tf, sf = await asyncio.gather(concurrent_target_features(c, w, h, tf), concurrent_style_features(c, w, h, sf))
            

            style_loss += torch.mean((tf - sf) ** 2 / (c * h * w))

        total_loss = alpha * base_loss + beta * style_loss
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

