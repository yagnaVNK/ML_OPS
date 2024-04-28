import logging
from zenml import step
from torch.utils.data import DataLoader, Dataset
from src.HAE import *
from src.HQA import *
from src.efficientNet_classifer import *
from tqdm import tqdm
from torchsig.utils.cm_plotter import plot_confusion_matrix
from sklearn.metrics import classification_report
from src.utils import *
from zenml.client import Client
import lightning.pytorch as pl
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings

mlflow_settings = MLFlowExperimentTrackerSettings(
    nested=True,
    tags={"key": "value"}
)

experiment_tracker = Client().active_stack.experiment_tracker
print(experiment_tracker)


@step(enable_cache=True,enable_artifact_visualization=True, experiment_tracker =experiment_tracker.name,
      settings={
        "experiment_tracker.mlflow": mlflow_settings
    })
def eval_HAE(classes: list,model: HAE,classifier: ExampleNetwork,ds_test: ModulationsDataset) -> list:
    """
    
    """
    accuracies = []
    num_recons = 1
    num_classes = len(classes)
    layers = len(model)
    mlflow.set_experiment("training_pipeline")
    run_name = f"HAE Evaluation - {num_classes} Classes"
    num_test_examples = len(ds_test)
    with mlflow.start_run(run_name=run_name,nested=True):
        for j in range(layers):
            for k in range(num_recons): 
                y_raw_preds = np.empty((num_test_examples, num_classes))
                y_preds = np.zeros((num_test_examples,))
                y_true = np.zeros((num_test_examples,))
                hae = model[j]
                hae = hae.float().to(device)
                hae.eval()
                classifier.to(device).eval()
                for i in tqdm(range(0, num_test_examples)):
                    # Retrieve data
                    idx = i  # Use index if evaluating over full dataset
                    
                    data, label = ds_test[idx]
                    #test_x = hae.reconstruct(data)
                    test_x = hae.reconstruct(torch.from_numpy(np.expand_dims(data, 0)).float().to(device))
                    #test_x = torch.from_numpy(np.expand_dims(data, 0)).float().to(device)
                    # Infer
                    #test_x = torch.from_numpy(np.expand_dims(test_x, 0)).float().to(device)
                    pred_tmp = classifier.predict(test_x)
                    pred_tmp = pred_tmp.cpu().numpy() if torch.cuda.is_available() else pred_tmp
                    # Argmax
                    y_preds[i] = np.argmax(pred_tmp)
                    # Store label
                    y_true[i] = label
            
            
                acc = np.sum(np.asarray(y_preds) == np.asarray(y_true)) / len(y_true)
                mlflow.log_metric(f"accuracy_layer_{j+1}", acc*100, step=j+1)
                plot_confusion_matrix(
                    y_true,
                    y_preds,
                    classes=classes,
                    normalize=True,
                    title="Example Modulations Confusion Matrix\nTotal Accuracy: {:.2f}%".format(
                        acc * 100
                    ),
                    text=False,
                    rotate_x_text=60,
                    figsize=(10, 10),
                )
                confusionMatrix_save_path = f"./vis/confusion_matrix_layer_{j+1}_recon_{k+1}.png"
                plt.savefig(confusionMatrix_save_path)
                mlflow.log_artifact(confusionMatrix_save_path)
                print(f"Layer {j+1}\nClassification Report: \nAccuracy {acc*100}")
                print(classification_report(y_true, y_preds))
                matplotlib.pyplot.close()
                accuracies.append(acc*100)


        fig, ax = plt.subplots()
        ax.plot(range(1, layers+1), accuracies, marker='o')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy per Layer")
        ax.grid(True)
        
        # Save the plot as an artifact
        plot_path = "./vis/accuracy_per_layer.png"
        fig.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        plt.close(fig)
    return accuracies

@step(enable_cache=False,enable_artifact_visualization=True,experiment_tracker =  experiment_tracker.name,
      settings={
        "experiment_tracker.mlflow": mlflow_settings
    })
def eval_HQA(classes: list, model: HQA, classifier: ExampleNetwork, ds_test: ModulationsDataset) -> list:
    accuracies = []
    num_recons = 1
    num_classes = len(classes)
    layers = len(model)
    mlflow.set_experiment("training_pipeline")
    run_name = f"HQA Evaluation - {num_classes} Classes"
    num_test_examples = len(ds_test)
    
    with mlflow.start_run(run_name=run_name, nested=True):
        for j in range(layers):
            for k in range(num_recons):
                y_raw_preds = np.empty((num_test_examples, num_classes))
                y_preds = np.zeros((num_test_examples,))
                y_true = np.zeros((num_test_examples,))
                hqa = model[j]
                hqa = hqa.float().to(device)
                hqa.eval()
                classifier.to(device).eval()

                # Initialize codebook usage count for each class
                codebook_usage_per_class = [torch.zeros(hqa.codebook.codebook_slots, device=device) for _ in range(num_classes)]

                for i in tqdm(range(0, num_test_examples)):
                    # Retrieve data
                    idx = i  # Use index if evaluating over full dataset
                    data, label = ds_test[idx]
                    test_x = hqa.reconstruct(torch.from_numpy(np.expand_dims(data, 0)).float().to(device))

                    # Infer
                    pred_tmp = classifier.predict(test_x)
                    pred_tmp = pred_tmp.cpu().numpy() if torch.cuda.is_available() else pred_tmp

                    # Argmax
                    y_preds[i] = np.argmax(pred_tmp)

                    # Store label
                    y_true[i] = label

                    # Update codebook usage count for the corresponding class
                    z_e = hqa.encode(test_x)
                    _, indices, _, _ = hqa.codebook(z_e)
                    indices_onehot = F.one_hot(indices, num_classes=hqa.codebook.codebook_slots).float()
                    codebook_usage_per_class[label] += indices_onehot.sum(dim=(0, 1))

                acc = np.sum(np.asarray(y_preds) == np.asarray(y_true)) / len(y_true)
                mlflow.log_metric(f"accuracy_layer_{j+1}", acc*100, step=j+1)

                # Plot codebook usage histogram for each class
                for class_idx in range(num_classes):
                    class_name = classes[class_idx]
                    codebook_usage = codebook_usage_per_class[class_idx].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(range(len(codebook_usage)), codebook_usage)
                    ax.set_xlabel('Codebook Index')
                    ax.set_ylabel('Usage Count')
                    ax.set_title(f'Codebook Usage Histogram - Layer {j+1} - Class {class_name}')

                    hist_save_path = f"./vis/codebook_usage_layer_{j+1}_class_{class_name}.png"
                    fig.savefig(hist_save_path)
                    mlflow.log_artifact(hist_save_path)
                    plt.close(fig)

                plot_confusion_matrix(
                    y_true,
                    y_preds,
                    classes=classes,
                    normalize=True,
                    title="Example Modulations Confusion Matrix\nTotal Accuracy: {:.2f}%".format(
                        acc * 100
                    ),
                    text=False,
                    rotate_x_text=60,
                    figsize=(10, 10),
                )
                confusionMatrix_save_path = f"./vis/confusion_matrix_layer_{j+1}_recon_{k+1}.png"
                plt.savefig(confusionMatrix_save_path)
                mlflow.log_artifact(confusionMatrix_save_path)
                print(f"Layer {j+1}\nClassification Report: \nAccuracy {acc*100}")
                print(classification_report(y_true, y_preds))
                matplotlib.pyplot.close()
    
                accuracies.append(acc*100)

        fig, ax = plt.subplots()
        ax.plot(range(1, layers+1), accuracies, marker='o')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy per Layer")
        ax.grid(True)

        # Save the plot as an artifact
        plot_path = "./vis/accuracy_per_layer.png"
        fig.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close(fig)

        index = 60
        model = hqa[0]
        # Retrieve z_q from the codebook
        z_q = model.codebook.lookup(torch.tensor(index).to(device)).squeeze()
    
        # Decode
        reconstructed_signal = model.decode(z_q).squeeze().detach().cpu().numpy()
    
        # Plot constellation diagram
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(reconstructed_signal.real, reconstructed_signal.imag, color='blue', alpha=0.8, label='Reconstructed Signal')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title(f'Constellation Diagram - Codebook Index {index}')
        ax.legend()
        ax.grid(True)
    
        # Save the plot
        plot_path = f"./vis/constellation_diagram_codebook_index_{index}.png"
        fig.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close(fig)
    

    return accuracies



@step(enable_cache=True,enable_artifact_visualization=True,experiment_tracker =  experiment_tracker.name,
      settings={
        "experiment_tracker.mlflow": mlflow_settings
    })
def eval_classifier(classes: list,classifier: ExampleNetwork,ds_test: ModulationsDataset) -> list:
    """
    
    """
    accuracies = []
    num_recons = 1
    num_classes = len(classes)
    
    mlflow.set_experiment("training_pipeline")
    run_name = f"classifier evaluation - {num_classes} Classes"
    num_test_examples = len(ds_test)
    with mlflow.start_run(run_name=run_name,nested=True):
        y_raw_preds = np.empty((num_test_examples, num_classes))
        y_preds = np.zeros((num_test_examples,))
        y_true = np.zeros((num_test_examples,))
        classifier.to(device).eval()
        for i in tqdm(range(0, num_test_examples)):
            # Retrieve data
            idx = i  # Use index if evaluating over full dataset
            
            data, label = ds_test[idx]
            test_x = torch.from_numpy(np.expand_dims(data, 0)).float().to(device)
            pred_tmp = classifier.predict(test_x)
            pred_tmp = pred_tmp.cpu().numpy() if torch.cuda.is_available() else pred_tmp
            # Argmax
            y_preds[i] = np.argmax(pred_tmp)
            # Store label
            y_true[i] = label
    
    
        acc = np.sum(np.asarray(y_preds) == np.asarray(y_true)) / len(y_true)
        mlflow.log_metric(f"accuracy", acc*100, step=1)
        plot_confusion_matrix(
            y_true,
            y_preds,
            classes=classes,
            normalize=True,
            title="Example Modulations Confusion Matrix\nTotal Accuracy: {:.2f}%".format(
                acc * 100
            ),
            text=False,
            rotate_x_text=60,
            figsize=(10, 10),
        )
        confusionMatrix_save_path = f"./vis/confusion_matrix_layer.png"
        plt.savefig(confusionMatrix_save_path)
        mlflow.log_artifact(confusionMatrix_save_path)
        print(f"\nClassification Report: \nAccuracy {acc*100}")
        print(classification_report(y_true, y_preds))
        matplotlib.pyplot.close()
        accuracies.append(acc*100)

    return accuracies