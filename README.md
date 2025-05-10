# Butterworth vs. Savitzky-Golay Filters

Scripts to compare the Butterworth vs. Savitzky-Golay filters in the frequency- and time-domains.

These scripts created the plots for [my blog post on filters for scientific post-processing](
https://mvernacc.github.io/portfolio/engineering_notes/filters_savgol_vs_butter).

## Installation

This project uses the `uv` python package manager. To run the scripts, first [install uv](https://docs.astral.sh/uv/#installation).

Then run:

```bash
uv run bode.py
```

The first time you run the script, uv will automatically create a new virtual environment and install the project's dependencies.
