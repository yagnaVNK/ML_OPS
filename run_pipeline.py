from pipelines.training_pipeline import training_pipeline

if __name__  == "__main__":
    classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
    training_pipeline(classes,1024,1000,True,16)