import torch
import mlflow
from zenml import step
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import classification_report
from src.HAE import HAE
from src.HQA import HQA
from zenml.client import Client
from src.efficientNet_classifer import ExampleNetwork
from torchsig.utils.cm_plotter import plot_confusion_matrix
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings
import matplotlib.pyplot as plt
import numpy as np
from src.Adverserial_Dataset import AdversarialModulationsDataset as Adv_Dataset
from torchsig.datasets.modulations import ModulationsDataset
import logging
from src.efficientNet_classifer import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow_settings = MLFlowExperimentTrackerSettings(
    nested=True,
    tags={"key": "value"}
)


experiment_tracker = Client().active_stack.experiment_tracker
print(experiment_tracker)

@step(enable_cache=False, enable_artifact_visualization=True, experiment_tracker=experiment_tracker.name,
      settings={
        "experiment_tracker.mlflow": mlflow_settings
    })
def create_adversarial_dataset_step(original_dataset: ModulationsDataset, classifier_model: ExampleNetwork, epsilon: float) -> Adv_Dataset:
    try:
        # Log the epsilon value to MLflow
        mlflow.log_param("epsilon",str(epsilon))
        adv_dataset = Adv_Dataset(original_dataset=original_dataset, classifier_model=classifier_model, epsilon=epsilon)
        print("created dataset")

        return adv_dataset
    except Exception as e:
        logger.error(f"Error in create_adversarial_dataset_step: {str(e)}", exc_info=True)
        raise
