![Example of the set](https://i.imgur.com/Pve0Vw0.png)

# Maldebroda set on Rust
Visualization of Mandelbrot set on Rust in zoom video.

Features:
- Multi-threaded rendering
- Option to customize the color of the fractal

## Usage
Generate frames in `output/`:
```bash
$ cargo run
```
Merge the frames into one video via `ffmpeg`:
```bash
$ ffmpeg -framerate 30 -pattern_type glob -i 'output/*.png' out.mp4
```
