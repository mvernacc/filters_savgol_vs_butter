def savgol_window_length(order: int, cutoff: float) -> int:
    """Calculate the window length for a Savitzky-Golay filter given a cutoff frequency.

    Args:
        order: Polynomial order
        cutoff: [dimensionless] Cutoff frequency / sample frequency
    
    References:
        [Schafer2011] R. W. Schafer, "On the frequency-domain properties of Savitzky-Golay filters,"
            in 2011 Digital Signal Processing and Signal Processing Education Meeting (DSP/SPE),
            Jan. 2011, pp. 54-59. doi: 10.1109/DSP-SPE.2011.5739186.
            Online: https://doi.org/10.1109/DSP-SPE.2011.5739186
    """
    fc = 2 * cutoff  # Schafer2011 defines fc relative to the Nyquist frequency, not the sample frequency
    m = int(round(((order + 1) / fc + 4.6) / 3.2))
    return 2 * m + 1