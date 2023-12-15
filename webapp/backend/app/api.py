from fastapi import FastAPI, Form , UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
from reportlab.lib.pagesizes import letter
from PIL import Image
import os
import shutil
from src.component.inference import Inference
from src.config.configuration import ConfigurationManager


app = FastAPI()

config_manager = ConfigurationManager()
model_config = config_manager.get_model_config()
inference_config = config_manager.get_inference_config()
inference = Inference(model_config = model_config, inference_config=inference_config)

class TemperatureManager:
    def __init__(self):
        self.temperature = 1  

temperature_manager = TemperatureManager()

origins = [
    "http://localhost:3000",
    "localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Get Route
@app.get("/", tags=["root"])
async  def read_root() -> dict:
    return {"message" : "Welcome to FastAPI!"}


@app.post("/download")
async def image(images: List[UploadFile] = File(...) , temperature: float = Form(...)):

    output_folder = "./imagesInference" 
    shutil.rmtree(output_folder, ignore_errors=False)  
    os.makedirs(output_folder, exist_ok=True)  

    temperature_manager.temperature = temperature

    for idx, uploaded_file in enumerate(images):
        image_path = os.path.join(output_folder, f"temp_image_{idx}.png")
        with open(image_path, "wb") as temp_image:
            shutil.copyfileobj(uploaded_file.file, temp_image)

    return {"images_saved": True}


@app.get("/captions")
async def return_captions():

    captions = {}

    output_folder = "./imagesInference"
    
    file_paths = os.listdir(output_folder)

    for file_path in file_paths:
        if os.path.isfile(os.path.join(output_folder,file_path)):
            image_path = os.path.join(output_folder,file_path)
            captions[file_path] =inference.generate_caption_batch(image_path, max_tokens=500, temperature=temperature_manager.temperature, deterministic=True)
            
    return {"captions" : captions}




