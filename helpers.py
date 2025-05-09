def savgol_window_length(order: int, cutoff: float) -> int:
    """Calculate the window length for a Savitzky-Golay filter given a cutoff frequency.

    Args:
        order: Polynomial order
        cutoff: [dimensionless] Cutoff frequency / sample frequency
    """
    # See https://doi.org/10.1109/DSP-SPE.2011.5739186, equation 11.
    fc = 2 * cutoff  # The paper defines fc relative to the Nyquist frequency, not the sample frequency
    m = int(round(((order + 1) / fc + 4.6) / 3.2))
    return 2 * m + 1