import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from src.logs import logger
from src.config.configuration import ConfigurationManager
from src.component.inference import Inference
from src.utils.commonFunctions import evaluate_model_generator


class InferencePipeline:
    def __init__(self):
        self.metrics: pd.DataFrame = pd.DataFrame(
            columns=['image_path', 'True Caption', 'Generated caption', 'rouge1', 'rouge2', 'rougeL', 'bleu', 'temperature', 'deterministic'])

    def run(self):
        config = ConfigurationManager()
        inference_config = config.get_inference_config()
        model_config = config.get_model_config()
        test_data = pd.read_csv(inference_config.inference_data_path, sep=';')
        inference = Inference(model_config, inference_config)
        row = 0
        for i in range(10):
            for temp in np.linspace(0.5, 1.5, 5):
                det = False
                test = test_data.iloc[i]
                test_image_path, test_image_caption = test.iloc[0], test.iloc[1]
                generated_caption = inference.generate_caption(
                    test_image_path, max_tokens=500, temperature=temp, deterministic=det)
                rouge1, rouge2, rougeL, bleu = evaluate_model_generator(
                    test_image_caption, generated_caption)
                self.metrics.loc[row] = [test_image_path, test_image_caption, generated_caption,
                                         rouge1, rouge2, rougeL, bleu, temp, det]
                row += 1
        self.metrics.to_csv(inference_config.metrics_path,
                            index=False, sep=';')
        logger.info(
            "Inference metrics saved successfully at {inference_config.metrics_path}")
