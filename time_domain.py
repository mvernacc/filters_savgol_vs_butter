import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, savgol_filter, sosfilt, sosfiltfilt

from helpers import savgol_window_length

rng = np.random.default_rng(230498)

N = 1000
t = np.linspace(0, 10.0, N)  # [s] time

# Lorentzian pulse
gamma = 0.5
t0 = 7.0
x_true = (gamma**2) / ((t - t0) ** 2 + (gamma**2))

fs = N / t[-1]  # [Hz] sampling frequency
fc = 0.01 * fs  # [Hz] filter cutoff frequency
x_noisy = x_true + rng.normal(0.0, 0.1, N)
ORDER = 5  # 5 probably is fine, note that applying the filter forward-backward
# effectively doubles the filter order.

sos = butter(ORDER, Wn=fc, fs=fs, output="sos")
x_butter_fore_back = sosfiltfilt(sos, x_noisy)
x_butter_fore = sosfilt(sos, x_noisy)

window_length = savgol_window_length(ORDER, fc / fs)
x_savgol = savgol_filter(x_noisy, window_length, ORDER)


fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

ax1.plot(t, x_true, color="black", linestyle="--", label="true", zorder=3)
ax1.plot(
    t,
    x_noisy,
    marker=".",
    linestyle="none",
    color="gray",
    label=f"noisy, $f_s$ = {fs:.1f} Hz",
)
ax1.plot(
    t,
    x_butter_fore_back,
    color="tab:blue",
    linewidth=3.0,
    label=f"butter forward-back, order={ORDER}, cutoff={fc:.1f} Hz",
)
ax1.plot(
    t,
    x_butter_fore,
    color="tab:red",
    label=f"butter forward-only, order={ORDER}, cutoff={fc:.1f} Hz",
)
ax1.plot(
    t,
    x_savgol,
    color="xkcd:tan",
    label=f"savgol window_length={window_length:d}, polyorder={ORDER:d}",
)

ax1.legend(loc="upper left")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Amplitude [a.u.]")
ax1.grid(which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

ax1.set_xlim((5.0, 6.0))
ax1.set_ylim((0.05, 0.25))
fig.savefig("example_time_domain_detail.png", dpi=200)

ax1.set_xlim((0.0, t[-1]))
ax1.set_ylim((-0.3, 1.3))
fig.savefig("example_time_domain.png", dpi=200)


# Caclulate and plot the Fourier transform of each filtered signal
fft_noisy = np.fft.rfft(x_noisy)
fft_butter_fore_back = np.fft.rfft(x_butter_fore_back)
fft_savgol = np.fft.rfft(x_savgol)
freq = np.fft.rfftfreq(N, d=1.0 / fs)

fig, ax2 = plt.subplots(1, 1, figsize=(8, 6))
ax2.semilogx(freq, 20 * np.log10(np.abs(fft_noisy)), color="gray", label="noisy")
ax2.semilogx(
    freq,
    20 * np.log10(np.abs(fft_butter_fore_back)),
    color="tab:blue",
    linewidth=3.0,
    label=f"butter fore-back, order={ORDER}, cutoff={fc:.1f} Hz",
)
ax2.semilogx(
    freq,
    20 * np.log10(np.abs(fft_savgol)),
    color="xkcd:tan",
    label=f"savgol window_length={window_length:d}, polyorder={ORDER:d}",
)
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Fourier amplitude [dB]")
ax2.axvline(fc, color="black")
ax2.set_xlim((0.1, fs / 2.0))
ax2.set_ylim((-50, 50))
ax2.grid(which="both", linestyle="--", linewidth=0.5)
ax2.legend(loc="lower left")
plt.tight_layout()
fig.savefig("example_frequency_domain.png", dpi=200)

plt.show()
