# fft-viz

![teaser_fft_viz](/assets/teaser_fft_viz.png)
![teaser_fir_viz](/assets/teaser_fir_viz.png)

A high-quality FFT visualization tool

## 1. Features

### 1.1 FFT visualization

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

### 1.2 FIR filter visualization

- Supported filters:
  - All pass
  - Ideal low-pass
  - Ideal high-pass
  - Butterworth low-pass
  - Butterworth high-pass
  - Chebyshev type I low-pass
  - Chebyshev type I high-pass
  - Chebyshev type II low-pass
  - Chebyshev type II high-pass

## 2. Requirements

- Python 3.11 or later
- (Optional) CUDA Toolkit 12.0 or later and NVIDIA GPU
  Used for GPU acceleration. If a compatible GPU is unavailable, the code will fall back to CPU execution with OpenMP

## 3. Installation

```bash
git clone https://github.com/ShineiArakawa/fft_viz.git
cd fft_viz

pip install -r requirements.txt

# Alternatively, if you are use 'uv' for dependency management:
# uv sync
```

## 4. Usage

Launch the main visualizer with:

```bash
cd fft_viz

python fft_viz.py

# Alternatively, if you are use 'uv' for dependency management:
# uv run fft_viz.py

# You can also run the FIR (finite impluse response) filter visualizer with:
# python fir_viz.py
# or
# uv run fir_viz.py
```

The first run may take up to ~20 seconds as it compiles C++/CUDA extensions. Subsequent runs will be much faster thanks to caching.

You can also visualize window functions or FIR filters solely with:

```bash
cd fft_viz

python -m lib.windowing

python -m lib.filtering
```

## 5. License

This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.
