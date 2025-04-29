from scipy.signal import firwin
import numpy as np


class FIR:
    def __init__(self, fs: float, fc: float, order: int = 100, is_lowpass: bool = True, window: str = 'hamming'):
        self.fs = fs
        self.fc = fc
        self.order = order
        self.window = window
        self.taps = None
        self.is_lowpass = is_lowpass
        self.generate()

    def generate(self):
        nyq = self.fs / 2
        normalized_cutoff = self.fc / nyq
        self.taps = firwin(numtaps=self.order + 1, cutoff=normalized_cutoff, window=self.window, pass_zero=self.is_lowpass)
        return self.taps
    
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        return np.convolve(data, self.taps, mode='same')