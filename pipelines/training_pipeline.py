from zenml import pipeline
from steps.evaluate_model import *
from steps.load_data import *
from steps.train_model import *

@pipeline()
def training_pipeline(classes: list,
                    iq_samples: int, 
                    samples_per_class: int,
                    Train: bool,
                    batch_size: int) -> None:
    #classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
    
    dl_train = getDataLoader(classes,iq_samples,1000,True,16)
    dl_val = getDataLoader(classes,iq_samples,100,True,16)
    train_model(dl_train,dl_val,10)
    eval_model(dl_val)
    