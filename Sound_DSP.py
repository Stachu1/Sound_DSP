import numpy as np, matplotlib.pyplot as plt, time
from scipy.io import wavfile
from scipy import signal
from scipy.signal import find_peaks
from FIR_taps_generator import FIR


SAMPLE_RATE = 44100
INT16_MAX = 32767



def gen_square(freq: float, durration: float = 5) -> np.ndarray:
    t = np.linspace(0, durration, int(SAMPLE_RATE * durration), endpoint=False)
    waveform = INT16_MAX * signal.square(2 * np.pi * freq * t)
    return waveform


def gen_sine(freq: float, durration: float = 5) -> np.ndarray:
    t = np.linspace(0, durration, int(SAMPLE_RATE * durration), endpoint=False)
    waveform = INT16_MAX * np.sin(2 * np.pi * freq * t)
    return waveform


def gen_noise(amplitude: float = 1, durration: float = 5) -> np.ndarray:
    waveform = np.random.normal(0, 1, SAMPLE_RATE * durration)
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
    

def envelope_detection(waveform: np.ndarray, cutoff: int = 50, order: int = 100) -> np.ndarray:
    waveform = abs(waveform)
    fir = FIR(SAMPLE_RATE, cutoff, order)
    return fir.apply(waveform)


def detect_peaks_with_prominence(data, prominence=None, distance=None):
    "35280 the distance is calculated by multipling the time of the template by the sample rate" 
    "(not needed for the case i tested when chenged to standard deviation from median)"
    peaks, properties = find_peaks(data, prominence=prominence, distance=distance)
    return peaks


def save_wav(waveform: np.ndarray, file_name: str = "wave.wav"):
    wavfile.write(f"sounds/{file_name}", SAMPLE_RATE, waveform.astype(np.int16))


def read_wav(file_name: str) -> np.ndarray:
    sr, waveform = wavfile.read(f"sounds/{file_name}")
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)
    return waveform.astype(np.float32)


def plot_wav(waveform: np.ndarray, peaks=None):
    time_axis = np.linspace(0, len(waveform)/SAMPLE_RATE, len(waveform))
    plt.plot(time_axis, waveform, label="Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.legend()
    plt.grid(True)

    if peaks is not None:
        plt.plot(time_axis[peaks], waveform[peaks], "rx", label="Detected Peaks") 


def spectogram(waveform: np.ndarray, freq_lim: int = 10000):
    plt.specgram(waveform, Fs=SAMPLE_RATE, NFFT=1024, noverlap=512, cmap="inferno")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")
    plt.colorbar(label="Intensity (dB)")
    plt.ylim(0, freq_lim)


def plot_fft(waveform: np.ndarray):
    fft_result = np.fft.fft(waveform)
    freqs = np.fft.fftfreq(len(waveform), 1/SAMPLE_RATE)
    magnitude = np.abs(fft_result[:len(waveform)//2])
    freqs = freqs[:len(waveform)//2]
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, magnitude)
    plt.title('FFT of the waveform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)


if __name__ == "__main__":
    fir_lowpass = FIR(SAMPLE_RATE, 5000, order=2000, fir_type="lowpass")
    fir_highpass = FIR(SAMPLE_RATE, 4000, order=2000, fir_type="highpass")


    t_start = time.time()
    print("Gathering waveforms...", end="\r")

    waveform = trim_wav(read_wav("Bird.wav"), 0, 10)
    waveform = fir_lowpass.apply(waveform)
    waveform = fir_highpass.apply(waveform)
    waveform = normalize(waveform, INT16_MAX)

    template = trim_wav(waveform, 1.75, 1.95)

    print(f"Gathering waveforms - \33[92mDone\33[0m in {time.time() - t_start:.2f}s\nApplying matched filter...", end="\r")
    t_start = time.time()

    correlation = matched_filter(waveform, template)

    print(f"Applying matched filter - \33[92mDone\33[0m in {time.time() - t_start:.2f}s\nApplying envelope detection...", end="\r")
    t_start = time.time()

    envelope = envelope_detection(correlation, 30, 600)

    print(f"Applying envelope detection - \33[92mDone\33[0m in {time.time() - t_start:.2f}s\nDetecting peaks...", end="\r")
    t_start = time.time()

    noise_level = np.std(envelope)
    peaks = detect_peaks_with_prominence(envelope, prominence=1.7 * noise_level)

    print(f"Detecting peaks - \33[92mDone\33[0m in {time.time() - t_start:.2f}s")

    # plot_wav(envelope, peaks)
    plot_wav(normalize(envelope), peaks)
    plt.show()
