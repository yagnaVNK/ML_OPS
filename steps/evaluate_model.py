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

@step(enable_cache=True,enable_artifact_visualization=True)
def eval_HAE(classes: list,model: HAE,classifier: ExampleNetwork,ds_test: ModulationsDataset) -> list:
    """
    
    """
    accuracies = []
    num_recons = 1
    num_classes = len(classes)
    layers = len(model)
    
    num_test_examples = len(ds_test)
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
            plot_confusion_matrix(
                y_true,
                y_preds,
                classes=classes,
                normalize=True,
                title="Example Modulations Confusion Matrix\nTotal Accuracy: {:.2f}%".format(
                    acc * 100
                ),
                text=False,
                rotate_x_text=90,
                figsize=(16, 9),
            )
            '''
            confusionMatrix_save_path = f"\"
            path = Path(confusionMatrix_save_path)
            if os.path.exists(path):
                pass
            else:
                path.mkdir(parents=True)
                #plt.savefig(path)
            curr_path = os.getcwd()
            os.chdir(path)
            plt.savefig(f'Layer_{j+1}_reconstruction_{k+1}.png')
            os.chdir(curr_path)
            '''
            print(f"Layer {j+1}\nClassification Report: \nAccuracy {acc*100}")
            print(classification_report(y_true, y_preds))
            matplotlib.pyplot.close()
            accuracies.append(acc*100)


    
    return accuracies

@step(enable_cache=True,enable_artifact_visualization=True)
def eval_HQA(classes: list,model: HQA,classifier: ExampleNetwork,ds_test: ModulationsDataset) -> list:
    """
    
    """
    accuracies = []
    num_recons = 1
    num_classes = len(classes)
    layers = len(model)
    data_transform = ST.Compose([
        ST.ComplexTo2D(),
    ])
    ds_test = ModulationsDataset(
        classes=classes,
        use_class_idx=True,
        level=0,
        num_iq_samples=1000,
        num_samples=int(num_classes*100),
        include_snr=False,
        transform = data_transform
    )

    num_test_examples = len(ds_test)
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
            plot_confusion_matrix(
                y_true,
                y_preds,
                classes=classes,
                normalize=True,
                title="Example Modulations Confusion Matrix\nTotal Accuracy: {:.2f}%".format(
                    acc * 100
                ),
                text=False,
                rotate_x_text=90,
                figsize=(16, 9),
            )
            '''
            confusionMatrix_save_path = f"\"
            path = Path(confusionMatrix_save_path)
            if os.path.exists(path):
                pass
            else:
                path.mkdir(parents=True)
                #plt.savefig(path)
            curr_path = os.getcwd()
            os.chdir(path)
            plt.savefig(f'Layer_{j+1}_reconstruction_{k+1}.png')
            os.chdir(curr_path)
            '''
            print(f"Layer {j+1}\nClassification Report: \nAccuracy {acc*100}")
            print(classification_report(y_true, y_preds))
            matplotlib.pyplot.close()
            accuracies.append(acc*100)

    return accuracies