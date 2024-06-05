![Example of the set](https://i.imgur.com/Pve0Vw0.png)

# Mandelbrot set on Rust
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
$ ffmpeg -framerate 24 -i output/fractal_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 -y video.mp4 
```
