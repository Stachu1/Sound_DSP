import numpy as np, matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

SAMPLE_RATE = 44100
DURRATION = 5
INT16_MAX = 32767




def gen_square(freq: float) -> np.ndarray:
    t = np.linspace(0, DURRATION, int(SAMPLE_RATE * DURRATION), endpoint=False)
    waveform = INT16_MAX * signal.square(2 * np.pi * freq * t)
    return waveform


def gen_sine(freq: float) -> np.ndarray:
    t = np.linspace(0, DURRATION, int(SAMPLE_RATE * DURRATION), endpoint=False)
    waveform = INT16_MAX * np.sin(2 * np.pi * freq * t)
    return waveform


def gen_noise(amplitude: float = 1) -> np.ndarray:
    waveform = np.random.normal(0, 1, SAMPLE_RATE * DURRATION)
    return normalize(waveform, amplitude)


def normalize(waveform: np.ndarray, amplitude: float = 1) -> np.ndarray:
    return amplitude * waveform / max(abs(waveform))


def mask_wav(waveform: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
    masked_waveform = np.zeros_like(waveform)
    start_sample = max(0, int(start_time * SAMPLE_RATE))
    end_sample = min(len(waveform), int(end_time * SAMPLE_RATE))
    masked_waveform[start_sample:end_sample] = waveform[start_sample:end_sample]
    return masked_waveform


def trim_wav(waveform: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
    start_sample = max(0, int(start_time * SAMPLE_RATE))
    end_sample = min(len(waveform), int(end_time * SAMPLE_RATE))
    return waveform[start_sample:end_sample]


def matched_filter(waveform: np.ndarray, template: np.ndarray) -> np.ndarray:    
    correlation = np.convolve(waveform, np.flip(template), mode='valid')
    return normalize(correlation)
    

def envelope_detection(waveform: np.ndarray, cutoff_freq: float = 10) -> np.ndarray:
    waveform = abs(waveform)
    cutoff_normalized = cutoff_freq / SAMPLE_RATE / 2.0
    b, a = signal.butter(4, cutoff_normalized, btype='low')
    return signal.filtfilt(b, a, waveform)


def save_wav(waveform: np.ndarray, file_name: str = "wave.wav") -> None:
    wavfile.write(file_name, SAMPLE_RATE, waveform.astype(np.int16))


def read_wav(file_name: str) -> np.ndarray:
    sr, waveform = wavfile.read(file_name)
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)
    return waveform.astype(np.float32)


def plot_wav(waveform: np.ndarray) -> None:
    time_axis = np.linspace(0, len(waveform)/SAMPLE_RATE, len(waveform))
    plt.plot(time_axis, waveform)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")


def spectogram(waveform: np.ndarray, freq_lim: int = 10000) -> None:
    plt.specgram(waveform, Fs=SAMPLE_RATE, NFFT=1024, noverlap=512, cmap="inferno")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")
    plt.colorbar(label="Intensity (dB)")
    plt.ylim(0, freq_lim)




waveform = read_wav("bird.wav")
waveform = trim_wav(waveform, 0, 5)
template = trim_wav(waveform, 2.45, 2.7)

correlation = matched_filter(waveform, template)
envelope = envelope_detection(correlation, 50)

plot_wav(envelope)
# plot_wav(waveform)
# spectogram(waveform)
plt.show()
