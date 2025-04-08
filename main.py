import numpy as np, matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt


SAMPLE_RATE = 44100
DURRATION = 31
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
    wavfile.write("sounds/" + file_name, SAMPLE_RATE, waveform.astype(np.int16))


def read_wav(file_name: str) -> np.ndarray:
    sr, waveform = wavfile.read("sounds/" + file_name)
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)
    return waveform.astype(np.float32)


def detect_peaks_with_prominence(data, prominence=None, distance=None):
    "35280 the distance is calculated by multipling the time of the template by the sample rate" 
    "(not needed for the case i tested when chenged to standard deviation from median)"
    peaks, properties = find_peaks(data, prominence=prominence, distance=distance)
    return peaks


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def plot_wav(waveform: np.ndarray, peaks: None) -> None:
    time_axis = np.linspace(0, len(waveform)/SAMPLE_RATE, len(waveform))
    plt.plot(time_axis, waveform, label="Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.legend()
    plt.grid(True)

    if peaks is not None:
        print("Detected Peaks at Times (seconds):")
        for peak in peaks:
            peak_time = peak / SAMPLE_RATE
            print(f"{peak_time:.6f} s")
        plt.plot(time_axis[peaks], waveform[peaks], "rx", label="Detected Peaks") 


def spectogram(waveform: np.ndarray, freq_lim: int = 10000) -> None:
    plt.specgram(waveform, Fs=SAMPLE_RATE, NFFT=1024, noverlap=512, cmap="inferno")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")
    plt.colorbar(label="Intensity (dB)")
    plt.ylim(0, freq_lim)


def plot_fft_of_template(template: np.ndarray):
    fft_result = np.fft.fft(template)
    freqs = np.fft.fftfreq(len(template), 1/SAMPLE_RATE)
    magnitude = np.abs(fft_result[:len(template)//2])
    freqs = freqs[:len(template)//2]
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, magnitude)
    plt.title('FFT of the Template (Trimmed Square Wave)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)



waveform = read_wav("Loud_bn_beep_different_volume.wav")
waveform = normalize(waveform, INT16_MAX)

template = trim_wav(read_wav("Car_beep_2.wav"), 1.0, 1.8)
template = normalize(template, INT16_MAX)

correlation = matched_filter(waveform, template)
envelope = envelope_detection(correlation, 50)

#peaks = detect_peaks(envelope, 0.25, min_distance=5000)
#peaks = detect_peaks_with_prominence(envelope,  0.18)

noise_level = np.std(envelope)
peaks = detect_peaks_with_prominence(envelope, prominence=2.7 * noise_level)

plot_wav(envelope, peaks)
plt.show()
