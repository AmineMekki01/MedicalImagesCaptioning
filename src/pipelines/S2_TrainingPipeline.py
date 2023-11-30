import pandas as pd
import torch
from src.logs import logger
from src.config.configuration import ConfigurationManager
from src.component.train import Trainer
from src.utils.commonFunctions import split_data, save_splitted_data
from src.component.chest_dataset import Dataset, collate_fn
from src.component.imageTransform import train_tfms, valid_tfms


class ModelTrainingPipeline:

    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        model_config = config.get_model_config()
        dataframe = pd.read_csv(training_config.training_data_path, sep=';')
        train, validation, test = split_data(dataframe)
        save_splitted_data(train, validation, test)
        train_dataset = Dataset(train, train_tfms)
        validation_dataset = Dataset(validation, valid_tfms)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_config.batch_size,
                                                       shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True, collate_fn=collate_fn)
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=training_config.batch_size, shuffle=False, pin_memory=True, num_workers=2, persistent_workers=True, collate_fn=collate_fn)

        trainer = Trainer(model_config, training_config,
                          (train_dataloader, validation_dataloader))
        trainer.fit()
        metrics = trainer.metrics
        metrics.to_csv(training_config.metrics_path, index=False, sep=';')
