from zenml import pipeline
from steps.evaluate_model import *
from steps.load_data import *
from steps.train_model import *
from steps.train_classifier import *
from typing import Tuple
from typing_extensions import Annotated
from steps.eval_adverserial_attack import *
from steps.adv_dataset import *



@pipeline(enable_artifact_metadata=True, enable_step_logs=True)
def NoiseTraining_pipeline():