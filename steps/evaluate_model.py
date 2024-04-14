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

experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=True,enable_artifact_visualization=True,experiment_tracker =  experiment_tracker.name)
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

@step(enable_cache=True,enable_artifact_visualization=True,experiment_tracker =  experiment_tracker.name)
def eval_HQA(classes: list,model: HQA,classifier: ExampleNetwork,ds_test: ModulationsDataset) -> list:
    """
    
    """
    accuracies = []
    num_recons = 1
    num_classes = len(classes)
    layers = len(model)
    
    mlflow.set_experiment("training_pipeline")
    run_name = f"HQA Evaluation - {num_classes} Classes"
    num_test_examples = len(ds_test)
    with mlflow.start_run(run_name=run_name,nested=True):
        for j in range(layers):
            for k in range(num_recons): 
                y_raw_preds = np.empty((num_test_examples, num_classes))
                y_preds = np.zeros((num_test_examples,))
                y_true = np.zeros((num_test_examples,))
                hqa = model[j]
                hqa = hqa.float().to(device)
                hqa.eval()
                classifier.to(device).eval()
                for i in tqdm(range(0, num_test_examples)):
                    # Retrieve data
                    idx = i  # Use index if evaluating over full dataset
                    
                    data, label = ds_test[idx]
                    #test_x = hae.reconstruct(data)
                    test_x = hqa.reconstruct(torch.from_numpy(np.expand_dims(data, 0)).float().to(device))
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