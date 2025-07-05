# fft-viz

![teaser_fft_vis](/assets/teaser_fft_vis.png)
![teaser_fir_vis](/assets/teaser_fir_vis.png)

A high-quality FFT visualization tool

## Features

- 2D DCT analysis on:
  - Sinusoidal wave image
  - Gray or RGB Gaussian noise image
  - Gray or RGB image from file
- Windowing in pixel-space to mitigate discontinuities on image boundaries
  - 10+ window functions are implemented
- Padding in pixel-space to make sure that the support has zero signals
- 2D power spectral density plot
- Line plots of radial, horizontal, and vertical power spectral density
- Interactive visualization on above features

## Requirements

- Python 3.11 or later
- (Optional) CUDA Toolkit 12.0 or later and NVIDIA GPU
  Used for GPU acceleration. If a compatible GPU is unavailable, the code will fall back to CPU execution with OpenMP

## Installation

```bash
git clone https://github.com/ShineiArakawa/colors-of-noise-2d.git
cd fft_viz

pip install -r requirements.txt

# Alternatively, if you are use 'uv' for dependency management:
# uv sync
```

## Usage

Launch the main visualizer with:

```bash
cd fft_viz

python interactive.py

# Alternatively, if you are use 'uv' for dependency management:
# uv run interactive.py
```

The first run may take up to ~20 seconds as it compiles C++/CUDA extensions. Subsequent runs will be much faster thanks to caching.

## License

This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.