import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, freqz_sos, savgol_coeffs, freqz

N = 1024


def freqz_butter_forward_back(order, cutoff):
    # Design Butterworth filter in second-order sections (sos)
    sos = butter(order, cutoff, btype="low", output="sos")
    w, h = freqz_sos(sos, worN=N)
    h_eff = h * np.conj(h)  # Is the backwards filter the complex conjugate of h?
    return w, h_eff, f"butter fore-back order={order:d}, cutoff={cutoff:.1f}"


def freqz_savgol(window_length, polyorder):
    coeffs = savgol_coeffs(window_length, polyorder, deriv=0)
    w, h = freqz(coeffs, worN=N)
    return w, h, f"savgol window_length={window_length:d}, polyorder={polyorder:d}"


# Plot Bode magnitude and phase response
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

for w, h, label in [
    freqz_butter_forward_back(3, 0.1),
    freqz_butter_forward_back(4, 0.1),
    freqz_savgol(21, 3),
    freqz_savgol(21, 4),
    freqz_savgol(21, 6),
    freqz_savgol(41, 3),
]:
    freqs = w / np.pi
    ax1.semilogx(freqs, 20 * np.log10(abs(h)), label=label)  # Magnitude response
    ax2.semilogx(freqs, np.angle(h, deg=True), label=label)  # Phase response

ax1.set_title("Bode Plot of Zero-Phase Butterworth vs. Savitzky-Golay Filters")
ax1.set_ylim((-100, 3))
ax1.set_ylabel("Magnitude (dB)")
ax1.grid(which="both", linestyle="--", linewidth=0.5)
ax1.legend(loc="lower left")

ax2.set_xlabel("Normalized Frequency (Nyquist = 1)")
ax2.set_ylabel("Phase (degrees)")
ax2.grid(which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.show()
