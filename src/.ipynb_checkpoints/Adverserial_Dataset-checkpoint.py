import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from typing import Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdversarialModulationsDataset(Dataset):
    def __init__(self, original_dataset, classifier_model, epsilon=0.1):
        self.original_dataset = original_dataset
        self.classifier_model = classifier_model
        self.epsilon = epsilon

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, torch.Tensor], Union[int, torch.Tensor]]:
        try:
            iq_sample, label = self._load_sample(idx)
            
            if isinstance(iq_sample, torch.Tensor):
                iq_sample = iq_sample.numpy()

            real_part = iq_sample[0, :]
            imaginary_part = iq_sample[1, :]

            amplitude = np.sqrt(real_part**2 + imaginary_part**2)
            angle = np.arctan2(imaginary_part, real_part)

            iq_sample_tensor = torch.from_numpy(iq_sample).unsqueeze(0).float()
            iq_sample_tensor.requires_grad = True
            
            #create iq amp tensor 
            output = self.classifier_model(iq_sample_tensor)
            loss = torch.sum(output)
            loss.backward()
    
            gradient_signs = torch.sign(iq_sample_tensor.grad.data)

            #epsilon based on the bits and samples per second.
            perturbation = self.epsilon * torch.where(
                (gradient_signs[0, 0, :] < 0) & (gradient_signs[0, 1, :] < 0),
                -gradient_signs,
                gradient_signs
            )

            attacked_amplitude = amplitude + perturbation[0, 0, :].numpy()

            attacked_real_part = attacked_amplitude * np.cos(angle)
            attacked_imaginary_part = attacked_amplitude * np.sin(angle)

            attacked_iq_sample = np.stack([attacked_real_part, attacked_imaginary_part], axis=0)

            if isinstance(self.original_dataset[idx][0], torch.Tensor):
                attacked_iq_sample = torch.from_numpy(attacked_iq_sample).float()

            return attacked_iq_sample, iq_sample, label

        except Exception as e:
            logger.error(f"Error in __getitem__ for index {idx}: {str(e)}", exc_info=True)
            raise

    def _load_sample(self, idx: int) -> Tuple[Union[np.ndarray, torch.Tensor], Union[int, torch.Tensor]]:
        try:
            if hasattr(self.original_dataset, 'file_paths'):
                file_path = self.original_dataset.file_paths[idx]
                data = np.load(file_path)
                iq_sample = data['iq_data']
                label = data['label']
            else:
                iq_sample, label = self.original_dataset[idx]

            return iq_sample, label
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}", exc_info=True)
            raise
