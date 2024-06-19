import pandas as pd
import numpy as np
from src.logs import logger
from src.config.configuration import ConfigurationManager
from src.component.inference import Inference
from src.utils.common_functions import evaluate_model_generator


class InferencePipeline:
    def __init__(self, batch_size=4):
        self.batch_size = batch_size
        self.metrics: pd.DataFrame = pd.DataFrame(
            columns=['image_path', 'True Caption', 'Generated caption', 'rouge1', 'rouge2', 'rougeL', 'bleu', 'cider_score', 'temperature', 'deterministic'])
        
    def run(self):
        config = ConfigurationManager()
        inference_config = config.get_inference_config()
        model_config = config.get_model_config()
        metrics_path = inference_config.metrics_path
        metrics_path = str(metrics_path).replace('.csv', f'_{model_config.encoder_type}.csv')
        test_data = pd.read_csv(inference_config.inference_data_path, sep=';')
        inference = Inference(model_config, inference_config)
        first_write = True 
        len_test_data = 16
        for start_idx in range(0, len_test_data, self.batch_size):
            if self.batch_size > len_test_data:
                self.batch_size = len_test_data
            end_idx = min(start_idx + self.batch_size, len_test_data)
            batch = test_data.iloc[start_idx:end_idx]
            image_paths = batch.iloc[:, 0].tolist()
            true_captions = batch.iloc[:, 1].tolist()

                    
            generated_captions = inference.generate_caption_batch(
                image_paths, max_tokens=500, temperature=1.0, deterministic=True)
            print(f"generated_captions  {generated_captions}")
            for i, (test_image_path, test_image_caption, generated_caption) in enumerate(zip(image_paths, true_captions, generated_captions)):
                rouge1, rouge2, rougeL, bleu, cider_score = evaluate_model_generator(
                    test_image_caption, generated_caption)
                cider_score = cider_score[0]
                print(f"Index: {i}")
                print(f"Image Path: {test_image_path}, Type: {type(test_image_path)}")
                print(f"True Caption: {test_image_caption}, Type: {type(test_image_caption)}")
                print(f"Generated Caption: {generated_caption}, Type: {type(generated_caption)}")
                print(f"ROUGE1: {rouge1}, Type: {type(rouge1)}")
                print(f"ROUGE2: {rouge2}, Type: {type(rouge2)}")
                print(f"ROUGEL: {rougeL}, Type: {type(rougeL)}")
                print(f"BLEU: {bleu}, Type: {type(bleu)}")
                print(f"CIDEr: {cider_score}, Type: {type(cider_score)}")
                print(f"Temperature: 1.0, Type: {type(1.0)}")
                print(f"Deterministic: True, Type: {type(True)}")
                if isinstance(generated_caption, list):
                    generated_caption = " ".join(generated_caption)
                self.metrics.loc[i] = [test_image_path, test_image_caption, generated_caption, rouge1, rouge2, rougeL, bleu, cider_score, 1.0, True]                    
            if first_write:
                self.metrics.to_csv(metrics_path, mode='w', index=False, header=True)
                first_write = False
            else:
                self.metrics.to_csv(metrics_path, mode='a', index=False, header=False)

            logger.info("Batch processed and metrics updated.")

        logger.info("Inference pipeline completed successfully.")

