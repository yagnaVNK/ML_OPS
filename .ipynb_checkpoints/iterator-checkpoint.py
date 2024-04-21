import subprocess
import time

argument_sets = []
argument_sets.append("--iq_samples 1024 --samples_per_class 8000 --batch_size 64 --Hae_epochs 10 --Hqa_epochs 20 --classifier_epochs 10 --layers 5 --input_feature_dim 2 --codebook_slots 128 --codeword_dim 64 --codebook_init normal --batch_norm 1 --reset_choice 1 --num_res_blocks 2 --cos_reset 1 --compress 2 --hae_lr 0.0004 --hqa_lr 0.0004 --hae_Cos_coeff 0.5 --hqa_Cos_coeff 0 --KL_coeff 0.1 --CL_coeff 0.005 --visual_dir ./codebooks/ --trainbool True")

for arguments in argument_sets: 
    command1 = f"python run_pipeline.py {arguments}"
    subprocess.run(command1, shell=True)