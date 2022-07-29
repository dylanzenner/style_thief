from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from typing import List


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

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post('/upload')
async def receive_file(base_image: bytes = File(), style_image: bytes = File()):
    # image1 = Image.open(io.BytesIO(base_image))
    # image2 = Image.open(io.BytesIO(images[1]))
    # image1.show()
    # image2.show()
    print(base_image, style_image)
    return {'upload status': 'complete'}
