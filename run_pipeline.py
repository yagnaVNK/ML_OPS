import argparse
from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

def parse_args():
    parser = argparse.ArgumentParser(description='Run the training pipeline with custom parameters.')
    parser.add_argument('--iq_samples', type=int, default=1024)
    parser.add_argument('--samples_per_class', type=int, default=8000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--Hae_epochs', type=int, default=10)
    parser.add_argument('--Hqa_epochs', type=int, default=20)
    parser.add_argument('--classifier_epochs', type=int, default=10)
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--input_feature_dim', type=int, default=2)
    parser.add_argument('--codebook_slots', type=int, default=128)
    parser.add_argument('--codeword_dim', type=int, default=64)
    parser.add_argument('--codebook_init', type=str, default='normal')
    parser.add_argument('--batch_norm', type=int, default=1)
    parser.add_argument('--reset_choice', type=int, default=1)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--cos_reset', type=int, default=1)
    parser.add_argument('--compress', type=int, default=2)
    parser.add_argument('--hae_lr', type=float, default=4e-4)
    parser.add_argument('--hqa_lr', type=float, default=4e-4)
    parser.add_argument('--hae_Cos_coeff', type=float, default=0.0)
    parser.add_argument('--hqa_Cos_coeff', type=float, default=0)
    parser.add_argument('--KL_coeff', type=float, default=0.1)
    parser.add_argument('--CL_coeff', type=float, default=0.005)
    parser.add_argument('--visual_dir', type=str, default='./codebooks/')
    parser.add_argument('--trainbool', type=bool, default=True)
    parser.add_argument('--epsilon', type=float, default=0.1)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    eff_net_PATH = f"./src/classifiers/{args.classifier_epochs}epochs_classifier.pt"
    enc_hidden_sizes=[16, 16, 32, 64, 128, 256]
    dec_hidden_sizes=[16, 64, 256, 512, 1024, 2048]
    classes = ["4ask","8pam","16psk","32qam_cross","2fsk","ofdm-256"]
    #classes = ["Anger","Disgust","Fear","Sadness","Neutral","Amusement","Inspiration","Joy","Tenderness"]
    training_pipeline(classes=classes,
                      iq_samples=args.iq_samples, 
                      enc_hidden_sizes=enc_hidden_sizes,
                      dec_hidden_sizes=dec_hidden_sizes,
                      samples_per_class=args.samples_per_class,
                      batch_size=args.batch_size,
                      Hae_epochs=args.Hae_epochs,
                      Hqa_epochs=args.Hqa_epochs,
                      classifier_epochs=args.classifier_epochs,
                      layers=args.layers,
                      input_feature_dim=args.input_feature_dim,
                      codebook_slots=args.codebook_slots,
                      codebook_init=args.codebook_init,
                      codeword_dim=args.codeword_dim,
                      reset_choice=args.reset_choice,
                      batch_norm=args.batch_norm,
                      num_res_blocks=args.num_res_blocks,
                      cos_reset=args.cos_reset,
                      compress=args.compress,
                      hae_lr=args.hae_lr,
                      hqa_lr=args.hqa_lr,
                      hae_Cos_coeff=args.hae_Cos_coeff,
                      hqa_Cos_coeff=args.hqa_Cos_coeff,
                      KL_coeff=args.KL_coeff,
                      CL_coeff=args.CL_coeff,
                      visual_dir=args.visual_dir,
                      eff_net_path=eff_net_PATH,
                      trainbool=args.trainbool,
                      epsilon=args.epsilon,
                      )
