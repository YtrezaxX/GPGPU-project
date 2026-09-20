# GPGPU Project - Background Subtraction

## Project Overview

This project implements a real-time background subtraction algorithm for video processing, designed to separate moving objects from a static background. It is structured as a GStreamer filter and features two implementations for performance comparison:
1.  **CPU Reference:** A standard C++ implementation.
2.  **GPU Optimized:** A CUDA-based implementation for high-performance parallel processing.

The core algorithm involves:
*   **Background Estimation:** Pixel-wise weighted reservoir sampling to maintain a background model.
*   **Motion Detection:** Computing differences between the current frame and the estimated background.
*   **Mask Cleaning:** Morphological operations (erosion/dilation) to remove noise.
*   **Hysteresis Thresholding:** Reconstructing object masks by propagating strong signals to weaker connected areas.

## Key Files

*   **`src/Compute.cpp`**: Contains the CPU implementation of the background subtraction algorithm (`compute_cpp`).
*   **`src/Compute.cu`**: Contains the CUDA implementation (`compute_cu`) and kernel code.
*   **`src/stream.cpp`**: The main entry point. Sets up the GStreamer pipeline, parses command-line arguments, and invokes the processing filter.
*   **`src/gstfilter.c`**: Implementation of the custom GStreamer element.
*   **`src/Image.hpp`**: Helper structures for image buffer management (`ImageView`, `Image`) and memory handling (allocating on host vs. device).
*   **`CMakeLists.txt`**: Build configuration, defining dependencies (GStreamer, CUDA) and targets.

## Building and Running

### Prerequisites
*   **CMake** (version 3.18+)
*   **C++ Compiler** supporting C++20
*   **CUDA Toolkit**
*   **GStreamer** development libraries (`gstreamer-1.0`, `gstreamer-video-1.0`)
*   **Nix** (Optional, flake provided for environment setup)

### Build Steps

1.  **Configure:**
    ```bash
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    ```
    *Use `Debug` for debugging symbols.*

2.  **Compile:**
    ```bash
    cmake --build build
    ```

### Execution

The main executable is `stream`. It takes an input video file and an optional execution mode (CPU or GPU).

**Syntax:**
```bash
./build/stream --mode=[gpu|cpu] <input_video.mp4> [--output=output.mp4]
```

**Examples:**

*   Run on CPU and save output:
    ```bash
    ./build/stream --mode=cpu samples/ACET.mp4 --output=output.mp4
    ```

*   Run on GPU (requires `compute_cu` implementation):
    ```bash
    ./build/stream --mode=gpu samples/ACET.mp4 --output=output_gpu.mp4
    ```

*   Run without saving (display only):
    ```bash
    ./build/stream --mode=cpu samples/ACET.mp4
    ```

## Development Conventions

*   **Language Standards:** C++20.
*   **Memory Management:** The project uses a "zero-copy" architecture where possible. The input buffer provided by GStreamer is modified in-place to produce the output visualization (differences highlighted in red).
*   **Image Structure:** `ImageView<rgb8>` is used to pass image data (buffer pointer, dimensions, stride) without ownership.
*   **Parameters:** Algorithm parameters (thresholds, reservoir size, etc.) are currently defined as macros/constants in `src/Compute.cpp`.
*   **CUDA:** CUDA code resides in `.cu` files. Kernel launches and device memory management should be handled within `src/Compute.cu`.
