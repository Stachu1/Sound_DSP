import numpy as np, matplotlib.pyplot as plt, time
from scipy.io import wavfile
from scipy import signal
from scipy.signal import find_peaks, fftconvolve
from FIR_taps_generator import FIR


SAMPLE_RATE = 44100



def gen_square(freq: float, durration: float = 5) -> np.ndarray:
    t = np.linspace(0, durration, int(SAMPLE_RATE * durration), endpoint=False)
    waveform = signal.square(2 * np.pi * freq * t)
    return waveform


def gen_sine(freq: float, durration: float = 5) -> np.ndarray:
    t = np.linspace(0, durration, int(SAMPLE_RATE * durration), endpoint=False)
    waveform = np.sin(2 * np.pi * freq * t)
    return waveform


def gen_noise(amplitude: float = 1, durration: float = 5) -> np.ndarray:
    waveform = np.random.normal(0, amplitude, SAMPLE_RATE * durration)
    return waveform


def zero_pad(waveform: np.ndarray, durration: float, end=True) -> np.ndarray:
    if end:
        return np.concatenate((waveform, np.zeros(SAMPLE_RATE*durration)), axis=0)
    return np.concatenate((np.zeros(SAMPLE_RATE*durration), waveform), axis=0)


def normalize(waveform: np.ndarray, amplitude: float = 1) -> np.ndarray:
    return amplitude * waveform / np.max(np.abs(waveform))


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
    correlation = fftconvolve(waveform, np.flip(template), mode="same")
    return correlation
    

def envelope_detection(waveform: np.ndarray, cutoff: int = 50, order: int = 100) -> np.ndarray:
    waveform = np.abs(waveform)
    fir = FIR(SAMPLE_RATE, cutoff, order)
    return fir.apply(waveform)


def detect_peaks(data, prominence=None, distance=None) -> np.ndarray:
    peaks, properties = find_peaks(data, prominence=prominence, distance=distance)
    return peaks


def save_wav(waveform: np.ndarray, file_name: str = "wave.wav") -> None:
    wavfile.write(f"sounds/{file_name}", SAMPLE_RATE, waveform.astype(np.int16))


def read_wav(file_name: str) -> np.ndarray:
    sr, waveform = wavfile.read(f"sounds/{file_name}")
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)
    return waveform.astype(np.float32)


def plot_wav(waveform: np.ndarray, peaks=None, label="Waveform") -> None:
    time_axis = np.linspace(0, len(waveform)/SAMPLE_RATE, len(waveform))
    plt.plot(time_axis, waveform, label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Template Correlation over Time")
    plt.legend()
    plt.grid(True)

    if peaks is not None:
        plt.plot(time_axis[peaks], waveform[peaks], "rx", label="Detected Peaks") 


def spectogram(waveform: np.ndarray, freq_lim: int = 10000) -> None:
    plt.specgram(waveform, Fs=SAMPLE_RATE, NFFT=1024, noverlap=512, cmap="inferno")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")
    plt.colorbar(label="Intensity (dB)")
    plt.ylim(0, freq_lim)


def plot_fft(waveform: np.ndarray) -> None:
    fft_result = np.fft.fft(waveform)
    freqs = np.fft.fftfreq(len(waveform), 1/SAMPLE_RATE)
    magnitude = np.abs(fft_result[:len(waveform)//2])
    freqs = freqs[:len(waveform)//2]
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, magnitude)
    plt.title("FFT of the waveform")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)


if __name__ == "__main__":
    FILENAME = "Meow.wav"
    
    waveform = normalize(trim_wav(read_wav(FILENAME), 0, 5))
    template = waveform.copy()
    waveform = np.concatenate((waveform, waveform), axis=0)
    waveform = np.concatenate((waveform, waveform), axis=0)
    waveform = np.concatenate((waveform, waveform), axis=0)
    
    waveform += mask_wav(gen_noise(0.316, 40), 0, 5)
    waveform += mask_wav(gen_noise(0.562, 40), 5, 10)
    waveform += mask_wav(gen_noise(1, 40), 10, 15)
    waveform += mask_wav(gen_noise(1.778, 40), 15, 20)
    waveform += mask_wav(gen_noise(3.162, 40), 20, 25)
    waveform += mask_wav(gen_noise(5.623, 40), 25, 30)
    waveform += mask_wav(gen_noise(10, 40), 30, 35)
    waveform += mask_wav(gen_noise(17.783, 40), 35, 40)
    
    
    # save_wav(normalize(waveform, 32000), FILENAME.replace(".wav", "_input.wav"))
    
    # spectogram(waveform)
    # plot_wav(waveform, label="")
    # plt.title("Input Waveform")
    # plt.show()
    # exit(0)


    fir_lowpass = FIR(SAMPLE_RATE, 3000, order=1000, fir_type="lowpass")
    fir_highpass = FIR(SAMPLE_RATE, 1000, order=1000, fir_type="highpass")

    strat_time = time.time()
    d_time = time.time()
    print("Reading input file...", end="\r")

    # waveform = read_wav(FILENAME)

    print(f"Reading input file - \33[92mDone\33[0m in {time.time() - d_time:.2f}s\nFrequency filtering...", end="\r")
    d_time = time.time()

    # waveform = fir_lowpass.apply(waveform)
    # waveform = fir_highpass.apply(waveform)

    print(f"Frequency filtering - \33[92mDone\33[0m in {time.time() - d_time:.2f}s\nNormalizing & Storing template...", end="\r")
    d_time = time.time()

    # waveform = normalize(waveform)
    # template = trim_wav(waveform, 2, 3)

    print(f"Normalizing & Storing template - \33[92mDone\33[0m in {time.time() - d_time:.2f}s\nMatched filtering...", end="\r")
    d_time = time.time()

    correlation = matched_filter(waveform, template)

    print(f"Matched filtering - \33[92mDone\33[0m in {time.time() - d_time:.2f}s\nApplying envelope detection...", end="\r")
    d_time = time.time()

    envelope = envelope_detection(correlation, 10, 60000)

    print(f"Applying envelope detection - \33[92mDone\33[0m in {time.time() - d_time:.2f}s\nDetecting peaks...", end="\r")
    d_time = time.time()

    noise_level = np.std(envelope)
    peaks = detect_peaks(envelope, prominence=1.5 * noise_level)

    print(f"Detecting peaks - \33[92mDone\33[0m in {time.time() - d_time:.2f}s")
    print(f"\33[1;4mTotal processing time: {time.time() - strat_time:.2f}s\33[0m")

    envelope = normalize(envelope)
    # spectogram(waveform)
    plot_wav(envelope, peaks)
    plt.show()
    # save_wav(normalize(envelope * trim_wav(read_wav(FILENAME), 0, 60), 32767), FILENAME.replace(".wav", "_enhanced.wav"))
