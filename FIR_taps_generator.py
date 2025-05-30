from scipy.signal import firwin, fftconvolve
import numpy as np


class FIR:
    def __init__(self, fs: float, fc: float, order: int = 100, fir_type: str = "lowpass", window: str = "hamming"):
        self.fs = fs
        self.fc = fc
        self.order = order
        self.window = window
        self.taps = None
        self.fir_type = fir_type
        self.generate()


    def generate(self) -> np.ndarray:
        nyq = self.fs / 2
        normalized_cutoff = self.fc / nyq
        self.taps = firwin(numtaps=self.order + 1, cutoff=normalized_cutoff, window=self.window, pass_zero=self.fir_type)
        return self.taps


    def apply(self, data: np.ndarray) -> np.ndarray:
        return fftconvolve(data, self.taps, mode="same")
