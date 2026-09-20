# GPU Background Subtraction - GStreamer Plugin

A GStreamer filter that separates moving objects from the background in a video stream and
highlights the differences in red. Written as a CPU reference implementation in C++ and a
CUDA implementation, so the two can be compared frame for frame.

Background subtraction is a preliminary step in most video processing chains, and it is
almost entirely made of independent per-pixel operations - which makes it a good candidate
for the GPU.

## Pipeline

```
frame t -> background estimation -> change mask -> mask cleaning -> difference in red
              (persistent state)                    (morphology)
```

1. **Background estimation** (`bg_estimation.cu`) - a per-pixel online algorithm based on
   weighted reservoir sampling. Each pixel keeps K reservoir values with weights plus its own
   RNG state. For each new frame, the matching reservoir is found (colour distance below a
   threshold), weights and samples are updated, and the background becomes the colour with
   the highest weight. State persists across frames.
2. **Hysteresis threshold** (`threshold.cu`) - a double threshold (low 15, high 60) resolved
   by morphological reconstruction: strong pixels seed the markers, then the marker set is
   dilated under the weak-pixel mask until it stops growing. Keeps whole moving regions
   instead of only their strongest pixels.
3. **Mask cleaning** (`noise_deletion.cu`) - erosion followed by dilation, removing isolated
   pixels while keeping real moving regions.

## Optimisation work

The kernels exist in successive versions rather than a single final one, so the effect of
each optimisation can be measured in isolation:

- `bg_estimation_kernel_opti0` / `opti1` / `opti2` - naive version, then per-pixel RNG state
  handling, then memory access improvements.
- `reconstruction_opti0` / `opti1` - reconstruction over `bool` buffers versus packed `char`
  masks with a ping-pong index, avoiding a device-side read back per iteration.
- `noise_erosion_opti0` / `opti1` and `noise_diatation_opti0` / `noise_dilatation_opti1` -
  separate temporary buffer versus in-place processing.

`benchmark_versions.sh` checks out each tagged version in turn, rebuilds it, and times the
same video, so the comparison is reproducible:

```bash
./benchmark_versions.sh samples/video.mp4 3 results
```

## Build

0. If you're using Nix on the OpenStack, use the provided flake.

```
nix develop
```

1. Build the project (in Debug or Release) with cmake

```
export builddir=... # pas dans l'AFS
cmake -S . -B $builddir -DCMAKE_BUILD_TYPE=Debug
```

or

```
cmake -S . -B $builddir -DCMAKE_BUILD_TYPE=Release
```

2. Compile with Make:

```
make -C $builddir
```

3. Run with

```
$builddir/stream --mode=[gpu,cpu] <video.mp4> [--output=output.mp4]
```

`--mode=cpu` runs the C++ reference path, `--mode=gpu` runs the CUDA path. Same input, same
output format, so the two can be diffed directly.

4. Edit your cuda/cpp code in */Compute.*

## Layout

```
src/Compute.cpp        CPU reference implementation
src/Compute.cu         CUDA entry point, called once per frame
src/bg_estimation.cu   reservoir-sampling background model (3 versions)
src/threshold.cu       hysteresis threshold by reconstruction (2 versions)
src/noise_deletion.cu  erosion / dilation (2 versions)
src/gstfilter.c        GStreamer plugin wiring
src/stream.cpp         standalone runner
```

The plugin uses a zero-copy architecture: `compute_cu(ImageView<rgb8> in)` is called for each
frame and modifies `in` in place, which is what gets displayed or written out.
