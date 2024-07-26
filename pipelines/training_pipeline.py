from zenml import pipeline
from steps.evaluate_model import *
from steps.load_data import *
from steps.train_model import *
from steps.train_classifier import *
from typing import Tuple
from typing_extensions import Annotated


@pipeline(enable_artifact_metadata=True,enable_step_logs=True)
def training_pipeline(classes: list,
                    iq_samples: int, 
                    enc_hidden_sizes: list,
                    dec_hidden_sizes: list,
                    samples_per_class: int,
                    batch_size: int,
                    Hae_epochs: int,
                    Hqa_epochs: int,
                    classifier_epochs: int,
                    layers: int,
                    input_feature_dim: int,
                    codebook_slots: int,
                    codebook_init: str,
                    codeword_dim: int,
                    reset_choice: int,
                    batch_norm: int,
                    num_res_blocks: int,
                    cos_reset: int,
                    compress: int,
                    hae_lr: float,
                    hqa_lr: float,
                    hae_Cos_coeff: float,
                    hqa_Cos_coeff: float,
                    KL_coeff: float,
                    CL_coeff: float,
                    visual_dir: str,
                    eff_net_path: str,
                    trainbool: bool) -> Tuple[Annotated[list,"classifier_Accuracies"], Annotated[list,"HAE_Accuracies"], Annotated[list,"HQA_Accuracies"] ,Annotated[HAE,"hae_model"], Annotated[HQA,"hqa_model"] ]:

    '''
    
    dl_train,ds_train,dl_test,ds_test,dl_val,ds_val = getDataLoader(classes = classes,
                                                                    iq_samples = iq_samples,
                                                                    samples_per_class = samples_per_class,
                                                                    batch_size=batch_size)
    '''
    dl_train,ds_train,dl_test,ds_test,dl_val,ds_val = getDataLoader_EEG(classes=classes,data_dir=r"Processed_data",batch_size=batch_size)
    
    HAE_model = train_HAE(  dl_train = dl_train,
                            dl_val = dl_val,
                            epochs = Hae_epochs,
                            layers = layers,
                            enc_hidden_sizes= enc_hidden_sizes,
                            dec_hidden_sizes= dec_hidden_sizes,
                            input_feat_dim = input_feature_dim ,
                            codeword_dim = codeword_dim,
                            batch_norm = batch_norm,
                            num_res_blocks= num_res_blocks,
                            cos_reset = cos_reset,
                            compress= compress,
                            Cos_coeff = hae_Cos_coeff,
                            hae_lr=hae_lr)
    
    HQA_model = train_HQA(  dl_train = dl_train,
                            dl_val = dl_val,
                            epochs = Hqa_epochs,
                            layers = layers,
                            enc_hidden_sizes= enc_hidden_sizes,
                            dec_hidden_sizes= dec_hidden_sizes,
                            input_feat_dim = input_feature_dim,
                            codeword_dim = codeword_dim,
                            codebook_slots = codebook_slots,
                            codebook_init = codebook_init,
                            output_dir = visual_dir,
                            codeword_reset = reset_choice,
                            KL_coeff = KL_coeff,
                            CL_coeff = CL_coeff,
                            Cos_coeff = hqa_Cos_coeff,
                            batch_norm = batch_norm,
                            num_res_blocks = num_res_blocks,
                            cos_reset = cos_reset,
                            compress = compress,
                            model = HAE_model,
                            hqa_lr = hqa_lr)
    
    classifier = train_classifier(dl_train = dl_train,
                                  dl_val = dl_val,
                                  epochs = classifier_epochs,
                                  train_bool = trainbool,
                                  eff_net_PATH = eff_net_path,
                                  classes = classes,
                                  in_channels = input_feature_dim)
    
    accuracies_Hae = eval_HAE(classes,HAE_model,classifier,ds_test)
    
    accuracies_Hqa = eval_HQA(classes,HQA_model,classifier,ds_test)

    accuracies_classifier = eval_classifier(classes,classifier,ds_test)

    #generate_constellations(classes,HAE_model,HQA_model,ds_test)

    return accuracies_classifier, accuracies_Hae, None , HAE_model, None