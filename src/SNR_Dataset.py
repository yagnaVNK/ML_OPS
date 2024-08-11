from torchsig.datasets.modulations import ModulationsDataset
import numpy as np
from torchsig.transforms import (
    Compose,
    IQImbalance,
    Normalize,
    RandomApply,
    RandomFrequencyShift,
    RandomPhaseShift,
    RandomResample,
    RandomTimeShift,
    RayleighFadingChannel,
    TargetSNR,
)

class DualSNRModulationsDataset(ModulationsDataset):
    """DualSNRModulationsDataset generates pairs of signals, one with low SNR and one with high SNR, and normalizes them.
    
    Args:
        low_snr_range (tuple): The range of SNR values for the low SNR signal (e.g., (-2, 0)).
        high_snr_range (tuple): The range of SNR values for the high SNR signal (e.g., (20, 30)).
    """

    def __init__(
        self,
        low_snr_range: tuple = (-2, 0),
        high_snr_range: tuple = (20, 30),
        *args,
        **kwargs,
    ):
        super(DualSNRModulationsDataset, self).__init__(*args, **kwargs)
        self.low_snr_range = low_snr_range
        self.high_snr_range = high_snr_range
        self.normalization = Normalize(norm=np.inf)  # Normalization transform

    def __getitem__(self, item):
        # Get the base signal and its label
        signal, label = super(DualSNRModulationsDataset, self).__getitem__(item)
        
        # Apply low SNR to the signal
        low_snr_transform = TargetSNR(self.low_snr_range)
        low_snr_signal = low_snr_transform(signal)
        low_snr_signal = self.normalization(low_snr_signal)  # Normalize the low SNR signal
        
        # Apply high SNR to the signal
        high_snr_transform = TargetSNR(self.high_snr_range)
        high_snr_signal = high_snr_transform(signal)
        high_snr_signal = self.normalization(high_snr_signal)  # Normalize the high SNR signal
        
        return low_snr_signal, high_snr_signal, label
