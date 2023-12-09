import torch
from src.logs import logger
from src.config.configuration import ConfigurationManager
from src.component.train import Trainer
from src.utils.commonFunctions import read_train_val_csv
from src.component.dataset import Dataset, collate_fn
from src.component.imageTransform import train_tfms, valid_tfms


class ModelTrainingPipeline:

    def __init__(self):
        self.training_config = None
        self.ChestXray_training_config = None
        self.ROCO_training_config = None
        self.model_config = None

    def run_first_stage_fine_tuning(self):
        """
        Runs the first stage fine tuning pipeline for ROCO data.
        """
        train, validation = read_train_val_csv(
            self.ROCO_training_config.train_data_path, self.ROCO_training_config.val_data_path)
        train_dataset = Dataset(train, train_tfms)
        validation_dataset = Dataset(validation, valid_tfms)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.training_config.batch_size,
                                                       shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True, collate_fn=collate_fn)
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=self.training_config.batch_size, shuffle=False, pin_memory=True, num_workers=2, persistent_workers=True, collate_fn=collate_fn)

        trainer = Trainer(self.model_config, self.training_config, './artifacts/models/first_stage_model.pth',
                          (train_dataloader, validation_dataloader), use_pretrained=True)
        trainer.fit()
        trainer.save_model('./artifacts/models/first_stage_model.pth')
        metrics = trainer.metrics
        metrics.to_csv(self.ROCO_training_config.metrics_path,
                       index=False, sep=';')
    
    def run_second_stage_fine_tuning(self):
        """
        Runs the second stage fine tuning pipeline for ChestXray data.       
        """
        train, validation = read_train_val_csv(
            self.ChestXray_training_config.train_data_path, self.ChestXray_training_config.val_data_path)
        train_dataset = Dataset(train, train_tfms)
        validation_dataset = Dataset(validation, valid_tfms)
        logger.info(
            f"Running first stage fine tuning pipeline for ChestXray data")
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.training_config.batch_size,
                                                       shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True, collate_fn=collate_fn)
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=self.training_config.batch_size, shuffle=False, pin_memory=True, num_workers=2, persistent_workers=True, collate_fn=collate_fn)

        trainer = Trainer(self.model_config, self.training_config, './artifacts/models/second_stage_model.pth',
                          (train_dataloader, validation_dataloader), use_pretrained=True)

        trainer.load_best_model('./artifacts/models/first_stage_model.pth')
        trainer.fit()
        trainer.save_model('./artifacts/models/second_stage_model.pth')
        metrics = trainer.metrics
        metrics.to_csv(
            self.ChestXray_training_config.metrics_path, index=False, sep=';')

    def run_ChestXray(self):
        train, validation = read_train_val_csv(
            self.ChestXray_training_config.train_data_path, self.ChestXray_training_config.val_data_path)

        train_dataset = Dataset(train, train_tfms)
        validation_dataset = Dataset(validation, valid_tfms)
        logger.info(f"Running pipeline for ChestXray data")
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.training_config.batch_size,
                                                       shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True, collate_fn=collate_fn)
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=self.training_config.batch_size, shuffle=False, pin_memory=True, num_workers=2, persistent_workers=True, collate_fn=collate_fn)

        trainer = Trainer(self.model_config, self.training_config, './artifacts/models/ChestXray_model.pth',
                          (train_dataloader, validation_dataloader), use_pretrained=True)
        trainer.fit()
        trainer.save_model('./artifacts/models/ChestXray_model.pth')
        metrics = trainer.metrics
        metrics.to_csv(
            self.ChestXray_training_config.metrics_path, index=False, sep=';')

    def run(self):
        config = ConfigurationManager()
        self.model_config = config.get_model_config()
        training_configs = config.get_training_config()

        if len(training_configs) == 2:
            logger.info(f"Running pipeline for ChestXray data")
            self.training_config = training_configs[0]
            self.ChestXray_training_config = training_configs[1]
            self.run_ChestXray()

        elif len(training_configs) == 3:
            logger.info(
                f"Running pipeline for ROCO data and ChestXrayROCO data")
            self.training_config = training_configs[0]
            self.ChestXray_training_config = training_configs[1]
            self.ROCO_training_config = training_configs[2]
            self.run_first_stage_fine_tuning()
            self.run_second_stage_fine_tuning()

        else:
            logger.error("Error in training config")
