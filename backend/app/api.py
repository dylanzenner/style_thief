from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from typing import List
from starlette.responses import StreamingResponse
import cv2
import numpy as np



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


@app.post('/upload')
async def receive_file(base_image: bytes = File(), style_image: bytes = File()):
    image1 = Image.open(io.BytesIO(base_image))
    image2 = Image.open(io.BytesIO(style_image))
    
    bytes_image = io.BytesIO()
    image1.save(bytes_image, 'JPEG')
    bytes_image.seek(0)
    return StreamingResponse(content = bytes_image, media_type="image/jpeg")

    
    
    # return StreamingResponse(io.BytesIO(image1.tobytes()), media_type="image/jpeg")
