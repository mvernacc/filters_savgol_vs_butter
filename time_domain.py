"""This script compares Butterworth forward-backward and Savitzky-Golay filters
on a time-domain plot of an example signal.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, savgol_filter, sosfilt, sosfiltfilt

from helpers import savgol_window_length

rng = np.random.default_rng(230498)  # seed random number generator for repeatability

N = 1000  # number of measurements
t = np.linspace(0, 10.0, N)  # [s] time

# Create the true signal, a Lorentzian pulse.
gamma = 0.5
t0 = 7.0
x_true = (gamma**2) / ((t - t0) ** 2 + (gamma**2))  # [arbitrary units]
# Add noise to the signal.
x_noisy = x_true + rng.normal(0.0, 0.1, N)  # [arbitrary units]

# Set up the filters and filter the signal with each filter.
fs = N / t[-1]  # [Hz] sampling frequency
fc = 0.01 * fs  # [Hz] filter cutoff frequency
ORDER = 5  # 5 probably is fine, note that applying the filter forward-backward
# effectively doubles the filter order.

sos = butter(ORDER, Wn=fc, fs=fs, output="sos")
x_butter_fore_back = sosfiltfilt(sos, x_noisy)
x_butter_fore = sosfilt(sos, x_noisy)

window_length = savgol_window_length(ORDER, fc / fs)
x_savgol = savgol_filter(x_noisy, window_length, ORDER)

# Plot the true signal, noisy measurements, and filtered signals.
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
    color="tan",
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

plt.show()
