import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from typing import Tuple, Union
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdversarialModulationsDataset(Dataset):
    def __init__(self, original_dataset, epsilon=0.1):
        """
        Initialize the AdversarialModulationsDataset.

        Args:
            original_dataset: The original dataset to apply adversarial attacks on.
            epsilon (float): The magnitude of the adversarial perturbation.
        """
        self.original_dataset = original_dataset
        self.epsilon = epsilon
        

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, torch.Tensor], Union[int, torch.Tensor]]:
        try:
           
            
            # Load the original sample
            iq_sample, label = self._load_sample(idx)
            
            # Ensure iq_sample is a numpy array
            if isinstance(iq_sample, torch.Tensor):
                iq_sample = iq_sample.numpy()
            
            # Split the IQ sample into real and imaginary parts
            real_part = iq_sample[0, :]
            imaginary_part = iq_sample[1, :]
            
            # Calculate amplitude and angle
            amplitude = np.sqrt(real_part**2 + imaginary_part**2)
            angle = np.arctan2(imaginary_part, real_part)
            
            # Apply adversarial attack on amplitude
            attacked_amplitude = self._adversarial_attack_fn(amplitude)
            
            # Recalculate the real and imaginary parts
            attacked_real_part = attacked_amplitude * np.cos(angle)
            attacked_imaginary_part = attacked_amplitude * np.sin(angle)
            
            # Stack the attacked real and imaginary parts to form the IQ sample
            attacked_iq_sample = np.stack([attacked_real_part, attacked_imaginary_part], axis=0)
            
            # Convert back to torch.Tensor if the original was a Tensor
            if isinstance(self.original_dataset[idx][0], torch.Tensor):
                attacked_iq_sample = torch.from_numpy(attacked_iq_sample).float()
            
            return attacked_iq_sample, label
        except Exception as e:
            logger.error(f"Error in __getitem__ for index {idx}: {str(e)}", exc_info=True)
            raise

    def _load_sample(self, idx: int) -> Tuple[Union[np.ndarray, torch.Tensor], Union[int, torch.Tensor]]:
        """
        Load a sample from the original dataset.
        """
        try:
            if hasattr(self.original_dataset, 'file_paths'):
                # If the dataset has file paths, load from file
                file_path = self.original_dataset.file_paths[idx]
                data = np.load(file_path)
                iq_sample = data['iq_data']
                label = data['label']
            else:
                # Otherwise, use the __getitem__ method of the original dataset
                iq_sample, label = self.original_dataset[idx]
                
            return iq_sample, label
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}", exc_info=True)
            raise

    def _adversarial_attack_fn(self, amplitude: np.ndarray) -> np.ndarray:
        """
        Apply an adversarial attack to the amplitude.
        """
        try:
            perturbation = self.epsilon * np.sign(np.random.randn(*amplitude.shape))
            attacked_amplitude = amplitude + perturbation
            return attacked_amplitude
        except Exception as e:
            logger.error(f"Error in adversarial_attack_fn: {str(e)}", exc_info=True)
            raise