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
from sklearn.metrics import confusion_matrix

from src.Adverserial_Dataset import AdversarialModulationsDataset as Adv_Dataset

device = 'cuda'

mlflow_settings = MLFlowExperimentTrackerSettings(
    nested=True,
    tags={"key": "value"}
)


experiment_tracker = Client().active_stack.experiment_tracker
print(experiment_tracker)



def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, figsize=(10, 10), **kwargs):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Prevent division by zero
        cm = cm.astype(float) / row_sums
        cm = np.round(cm, 2)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Set tick locations and labels
    tick_marks = np.arange(len(classes))
    ax.set(xticks=tick_marks,
           yticks=tick_marks,
           xticklabels=classes,
           yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Ensure tick labels match the number of ticks
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    
    # Rotate and format the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.show()


@step(enable_cache=False,enable_artifact_visualization=True, experiment_tracker =experiment_tracker.name,
      settings={
        "experiment_tracker.mlflow": mlflow_settings
    })
def eval_three_models(classes: list, hae_model: HAE, hqa_model: HQA, classifier: ExampleNetwork, ds_test: Adv_Dataset) -> dict:
    """
    Evaluate HAE, HQA, and Classifier models on the adversarially attacked dataset.
    """
    results = {"HAE": [], "HQA": [], "Classifier": []}
    models = {"HAE": hae_model, "HQA": hqa_model, "Classifier": classifier}
    
    # Run evaluations for each model
    for model_name, model in models.items():
        accuracies = []
        num_classes = len(classes)
        num_test_examples = len(ds_test)
        
        mlflow.set_experiment("training_pipeline")
        run_name = f"{model_name} Evaluation - {num_classes} Classes"
        
        with mlflow.start_run(run_name=run_name, nested=True):
            y_preds = np.zeros((num_test_examples,))
            y_true = np.zeros((num_test_examples,))
            
            model = model.to(device).eval()
            for i in tqdm(range(0, num_test_examples)):
                # Retrieve data
                data, label = ds_test[i]
                data = torch.from_numpy(np.expand_dims(data, 0)).float().to(device)
                
                if model_name == "HAE":
                    output = model.reconstruct(data)
                elif model_name == "HQA":
                    output = model.reconstruct(data)
                elif model_name == "Classifier":
                    output = model.predict(data)
                
                # Prediction
                pred = output.cpu().detach().numpy() if torch.cuda.is_available() else output
                y_preds[i] = np.argmax(pred)
                y_true[i] = label
            
            # Calculate accuracy
            acc = np.sum(np.asarray(y_preds) == np.asarray(y_true)) / len(y_true)
            accuracies.append(acc * 100)
            
            mlflow.log_metric(f"{model_name}_accuracy", acc * 100)
            
            # Log confusion matrix
            plot_confusion_matrix(
                y_true,
                y_preds,
                classes=classes,
                normalize=True,
                title=f"{model_name} Confusion Matrix\nTotal Accuracy: {acc * 100:.2f}%",
                text=False,
                rotate_x_text=60,
                figsize=(10, 10),
            )

            confusionMatrix_save_path = f"./vis/confusion_matrix_{model_name}.png"
            plt.savefig(confusionMatrix_save_path)
            mlflow.log_artifact(confusionMatrix_save_path)
            plt.close()
            
            print(f"{model_name} Classification Report:\nAccuracy: {acc * 100:.2f}%")
            print(classification_report(y_true, y_preds))
            
        results[model_name] = accuracies

    return results
