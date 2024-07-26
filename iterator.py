import subprocess
import time

argument_sets = []

'''
argument_sets.append("--iq_samples 1024 --samples_per_class 8000 --batch_size 64 --Hae_epochs 10 --Hqa_epochs 20 --classifier_epochs 10 --layers 3 --input_feature_dim 2 --codebook_slots 64 --codeword_dim 128 --codebook_init normal --batch_norm 1 --reset_choice 1 --num_res_blocks 1 --cos_reset 1 --compress 2 --hae_lr 0.0004 --hqa_lr 0.0004 --hae_Cos_coeff 0.1 --hqa_Cos_coeff 0.1 --KL_coeff 0.1 --CL_coeff 0.005 --visual_dir ./codebooks/ --trainbool True")
argument_sets.append("--iq_samples 1024 --samples_per_class 8000 --batch_size 64 --Hae_epochs 10 --Hqa_epochs 20 --classifier_epochs 10 --layers 3 --input_feature_dim 2 --codebook_slots 64 --codeword_dim 128 --codebook_init normal --batch_norm 1 --reset_choice 1 --num_res_blocks 1 --cos_reset 1 --compress 2 --hae_lr 0.0004 --hqa_lr 0.0004 --hae_Cos_coeff 0.01 --hqa_Cos_coeff 0.01 --KL_coeff 0.1 --CL_coeff 0.005 --visual_dir ./codebooks/ --trainbool True")
argument_sets.append("--iq_samples 1024 --samples_per_class 8000 --batch_size 64 --Hae_epochs 10 --Hqa_epochs 20 --classifier_epochs 10 --layers 3 --input_feature_dim 2 --codebook_slots 64 --codeword_dim 128 --codebook_init normal --batch_norm 1 --reset_choice 1 --num_res_blocks 1 --cos_reset 1 --compress 2 --hae_lr 0.0004 --hqa_lr 0.0004 --hae_Cos_coeff 0.5 --hqa_Cos_coeff 0.5 --KL_coeff 0.1 --CL_coeff 0.005 --visual_dir ./codebooks/ --trainbool True")
argument_sets.append("--iq_samples 1024 --samples_per_class 8000 --batch_size 64 --Hae_epochs 10 --Hqa_epochs 20 --classifier_epochs 10 --layers 3 --input_feature_dim 2 --codebook_slots 64 --codeword_dim 128 --codebook_init normal --batch_norm 1 --reset_choice 1 --num_res_blocks 1 --cos_reset 1 --compress 2 --hae_lr 0.0004 --hqa_lr 0.0004 --hae_Cos_coeff 0.05 --hqa_Cos_coeff 0.05 --KL_coeff 0.1 --CL_coeff 0.005 --visual_dir ./codebooks/ --trainbool True")
argument_sets.append("--iq_samples 1024 --samples_per_class 8000 --batch_size 64 --Hae_epochs 10 --Hqa_epochs 20 --classifier_epochs 10 --layers 3 --input_feature_dim 2 --codebook_slots 64 --codeword_dim 128 --codebook_init normal --batch_norm 1 --reset_choice 1 --num_res_blocks 1 --cos_reset 1 --compress 2 --hae_lr 0.0004 --hqa_lr 0.0004 --hae_Cos_coeff 0.001 --hqa_Cos_coeff 0.001 --KL_coeff 0.1 --CL_coeff 0.005 --visual_dir ./codebooks/ --trainbool True")
argument_sets.append("--iq_samples 1024 --samples_per_class 8000 --batch_size 64 --Hae_epochs 10 --Hqa_epochs 20 --classifier_epochs 10 --layers 3 --input_feature_dim 2 --codebook_slots 64 --codeword_dim 128 --codebook_init normal --batch_norm 1 --reset_choice 1 --num_res_blocks 1 --cos_reset 1 --compress 2 --hae_lr 0.0004 --hqa_lr 0.0004 --hae_Cos_coeff 0.005 --hqa_Cos_coeff 0.005 --KL_coeff 0.1 --CL_coeff 0.005 --visual_dir ./codebooks/ --trainbool True")
'''
#argument_sets.append("--iq_samples 1024 --samples_per_class 8000 --batch_size 64 --Hae_epochs 10 --Hqa_epochs 20 --classifier_epochs 10 --layers 3 --input_feature_dim 2 --codebook_slots 128 --codeword_dim 128 --codebook_init normal --batch_norm 1 --reset_choice 1 --num_res_blocks 1 --cos_reset 1 --compress 2 --hae_lr 0.0004 --hqa_lr 0.0004 --hae_Cos_coeff 0 --hqa_Cos_coeff 0 --KL_coeff 0.1 --CL_coeff 0.005 --visual_dir ./codebooks/ --trainbool True")

#argument_sets.append("--iq_samples 1024 --samples_per_class 8000 --batch_size 64 --Hae_epochs 10 --Hqa_epochs 10 --classifier_epochs 10 --layers 3 --input_feature_dim 2 --codebook_slots 128 --codeword_dim 128 --codebook_init normal --batch_norm 1 --reset_choice 1 --num_res_blocks 0  --compress 2 --hae_lr 0.0004 --hqa_lr 0.0004  --CL_coeff 0 --visual_dir ./codebooks/ --trainbool True")

argument_sets.append(" --batch_size 16 --Hae_epochs 10 --Hqa_epochs 10 --classifier_epochs 30 --layers 1 --input_feature_dim 32 --codebook_slots 1024 --codeword_dim 1024 --codebook_init normal --batch_norm 0 --reset_choice 1 --num_res_blocks 4  --compress 2 --hae_lr 0.0004 --hqa_lr 0.0004  --CL_coeff 0 --visual_dir ./codebooks/ --trainbool True")


for arguments in argument_sets: 
    command1 = f"python run_pipeline.py {arguments}"
    subprocess.run(command1, shell=True)