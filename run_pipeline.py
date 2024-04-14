from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

if __name__  == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    iq_samples = 1024
    samples_per_class = 1000
    batch_size = 32
    Hae_epochs = 2
    Hqa_epochs = 2
    classifier_epochs = 15
    trainbool = False
    eff_net_PATH = f"./src/classifiers/{classifier_epochs}epochs_classifier.pt"

    layers = 2
    input_feature_dim = 2
    classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
    enc_hidden_sizes = [16, 16, 32, 64, 128, 256]
    dec_hidden_sizes = [16, 64, 256, 512, 1024, 2048]
    codebook_slots = 64
    codeword_dim = 64
    codebook_init = "normal"
    batch_norm = 1
    reset_choice = 1
    num_res_blocks = 0
    cos_reset = 1
    compress = 2
    hae_lr = 4e-4
    hqa_lr = 4e-4
    hae_Cos_coeff = 0.001
    hqa_Cos_coeff = 0.7
    KL_coeff = 0.1
    CL_coeff = 0.005

    training_pipeline(classes=classes,
                    iq_samples = iq_samples, 
                    enc_hidden_sizes = enc_hidden_sizes,
                    dec_hidden_sizes = dec_hidden_sizes,
                    samples_per_class = samples_per_class,
                    batch_size = batch_size,
                    Hae_epochs = Hae_epochs,
                    Hqa_epochs = Hqa_epochs,
                    classifier_epochs = classifier_epochs,
                    layers = layers,
                    input_feature_dim = input_feature_dim,
                    codebook_slots = codebook_slots,
                    codebook_init = codebook_init,
                    codeword_dim = codeword_dim,
                    reset_choice = reset_choice,
                    batch_norm = batch_norm,
                    num_res_blocks = num_res_blocks,
                    cos_reset = cos_reset,
                    compress = compress,
                    hae_lr = hae_lr,
                    hqa_lr = hqa_lr,
                    hae_Cos_coeff = hae_Cos_coeff,
                    hqa_Cos_coeff = hqa_Cos_coeff,
                    KL_coeff = KL_coeff,
                    CL_coeff = CL_coeff,
                    visual_dir = "./codebooks/",
                    eff_net_path = eff_net_PATH,
                    trainbool = trainbool)
    