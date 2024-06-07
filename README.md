![Example of the set](https://i.imgur.com/QJiQYlh.png)

# Mandelbrot set
Visualization of Mandelbrot set on NVIDIA CUDA.

Features:
- Multi-threaded rendering
- Native NVIDIA GPU support
- Coloring a fractal with a given gradient
- Zooming a fractal to a certain point by 10^15 times
- Anti-aliasing

Known Limitations:
- Images can only be generated with an aspect ratio of 1:1
- Scaling is limited by float64 accuracy

## Usage
Build the project. Make sure you have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
```bash
$ cargo build --release
```
Run program:
```bash
$ ./target/release/mandelbrot_set \
    --resolution 1024 \
    --colors "#00A3BC, #8B00BD, #81BD00, #BD5400" \
    --x -0.6582034218739634 \
    --y 0.44967917993930356 \
    --iters 5000 \
    --max-scale 1000000000000000 \
    --fps 24 \
    --seconds 180 \
    --n-samples 8 \
    --output ./output
```
Arguments:
- `--resolution` - resolution of the image
- `--colors` - gradient colors in hex format
- `--x` - x-coordinate for the zoom
- `--y` - y-coordinate for the zoom
- `--iters` - number of iterations
- `--max-scale` - maximum scale of the fractal
- `--fps` - frames per second
- `--seconds` - duration of the video
- `--n-samples` - number of samples for anti-aliasing
- `--output` - output directory for the frames

Merge the frames into one video via `ffmpeg`:
```bash
$ ffmpeg -framerate 24 -i output/frame_%09d.png -c:v libx264 -pix_fmt yuv420p -crf 18 -y video.mp4 
```
