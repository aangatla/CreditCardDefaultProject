from CreditCardDefault.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from CreditCardDefault.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact,DataTransformationArtifact
from CreditCardDefault.entity.config_entity import ModelTrainEvalConfig
from CreditCardDefault.exception import CreditCardDefaultException
import sys,os
from CreditCardDefault.logger import logging
from CreditCardDefault.components.data_ingestion import DataIngestion
from CreditCardDefault.components.data_validation import DataValidation
from CreditCardDefault.components.data_transformation import DataTransformation
from CreditCardDefault.components.model_train_eval import ModelTrainEval

class TrainPipeline:

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data ingestion")
            self.data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            self.data_ingestion_artifact = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {self.data_ingestion_artifact}")
            return self.data_ingestion_artifact
        except Exception as e:
            raise CreditCardDefaultException(e, sys)

    def start_data_validaton(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            self.data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            self.data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=self.data_validation_config
                                             )
            self.data_validation_artifact = self.data_validation.initiate_data_validation()
            return self.data_validation_artifact
        except  Exception as e:
            raise CreditCardDefaultException(e, sys)

    def start_data_transformation(self, data_ingestion_artifact:DataIngestionArtifact):

        try:
            self.data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config)
            self.data_transformation = DataTransformation(data_ingestion_artifact = data_ingestion_artifact,
                                                     data_transformation_config=self.data_transformation_config

                                                     )
            self.data_transformation_artifact = self.data_transformation.initiate_data_transformation()
            return self.data_transformation_artifact
        except  Exception as e:
            raise CreditCardDefaultException(e, sys)

    def start_model_train_eval(self, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_train_eval_config = ModelTrainEvalConfig(training_pipeline_config=self.training_pipeline_config)
            self.model_train_eval = ModelTrainEval(self.model_train_eval_config, data_transformation_artifact)
            self.model_train_eval_artifact = self.model_train_eval.initiate_model_train_eval()
            return self.model_train_eval_artifact
        except  Exception as e:
            raise CreditCardDefaultException(e, sys)


    def run_pipeline(self):
        try:

            self.data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            self.data_validation_artifact = self.start_data_validaton(
                data_ingestion_artifact=self.data_ingestion_artifact)
            self.data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=self.data_ingestion_artifact)
            model_train_eval_artifact = self.start_model_train_eval(self.data_transformation_artifact)

        except Exception as e:
            raise CreditCardDefaultException(e, sys)

