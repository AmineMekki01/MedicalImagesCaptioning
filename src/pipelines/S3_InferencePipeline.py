import pandas as pd
import numpy as np
from src.logs import logger
from src.config.configuration import ConfigurationManager
from src.component.inference import Inference
from src.utils.commonFunctions import evaluate_model_generator


class InferencePipeline:
    def __init__(self, batch_size=4):
        self.batch_size = batch_size
        self.metrics: pd.DataFrame = pd.DataFrame(
            columns=['image_path', 'True Caption', 'Generated caption', 'rouge1', 'rouge2', 'rougeL', 'bleu', 'temperature', 'deterministic'])
        
    def run(self):
        config = ConfigurationManager()
        inference_config = config.get_inference_config()
        model_config = config.get_model_config()
        test_data = pd.read_csv(inference_config.inference_data_path, sep=';')
        inference = Inference(model_config, inference_config)
        first_write = True 
        len_test_data = len(test_data)
        for start_idx in range(0, len_test_data, self.batch_size):
            if self.batch_size > len_test_data:
                self.batch_size = len_test_data
            end_idx = min(start_idx + self.batch_size, len_test_data)
            batch = test_data.iloc[start_idx:end_idx]
            image_paths = batch.iloc[:, 0].tolist()
            true_captions = batch.iloc[:, 1].tolist()

            for temp in np.linspace(0.5, 1.5, 5):
                for det in [True, False]:
                    
                    generated_captions = inference.generate_caption_batch(
                        image_paths, max_tokens=500, temperature=temp, deterministic=det)

                    for i, (test_image_path, test_image_caption, generated_caption) in enumerate(zip(image_paths, true_captions, generated_captions)):
                        rouge1, rouge2, rougeL, bleu = evaluate_model_generator(
                            test_image_caption, generated_caption)
                        self.metrics.loc[i] = [test_image_path, test_image_caption, generated_caption, rouge1, rouge2, rougeL, bleu, temp, det]                    
                    if first_write:
                        self.metrics.to_csv(inference_config.metrics_path, mode='w', index=False, header=True)
                        first_write = False
                    else:
                        self.metrics.to_csv(inference_config.metrics_path, mode='a', index=False, header=False)

            logger.info("Batch processed and metrics updated.")

        logger.info("Inference pipeline completed successfully.")

