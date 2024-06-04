use std::fs;
use image::{Rgb, RgbImage};
use rayon::prelude::*;
use num::complex::Complex;
use std::time::{Instant};

fn hex2rgb(hex: &str) -> Result<Vec<u8>, String> {
    let hex = hex.trim_start_matches('#');
    if hex.len() != 6 {
        return Err("Invalid HEX color length".to_string());
    }

    let r = u8::from_str_radix(&hex[0..2], 16).map_err(|_| "Invalid HEX color")?;
    let g = u8::from_str_radix(&hex[2..4], 16).map_err(|_| "Invalid HEX color")?;
    let b = u8::from_str_radix(&hex[4..6], 16).map_err(|_| "Invalid HEX color")?;

    Ok(vec![r, g, b])
}

fn lerp_color(color1: &[u8; 3], color2: &[u8; 3], value: f64) -> [u8; 3] {
    [
        (color1[0] as f64 + (color2[0] as f64 - color1[0] as f64) * value) as u8,
        (color1[1] as f64 + (color2[1] as f64 - color1[1] as f64) * value) as u8,
        (color1[2] as f64 + (color2[2] as f64 - color1[2] as f64) * value) as u8,
    ]
}

fn get_gradient(gradient_colors: Vec<&str>, max_iters: u32) -> Vec<[u8; 3]> {
    let mut colors = vec![];
    let mut gradient_colors_rgb = vec![];
    for color in &gradient_colors {
        let rgb = hex2rgb(color).unwrap();
        gradient_colors_rgb.push([rgb[0], rgb[1], rgb[2]]);
    }

    for i in 0..max_iters {
        let color_index = (i as usize * (gradient_colors.len() - 1)) / max_iters as usize;
        let color_value = (i as f64 * (gradient_colors.len() as f64 - 1.0)) / max_iters as f64;
        let value = color_value % 1.0;
        colors.push(lerp_color(
            &gradient_colors_rgb[color_index],
            &gradient_colors_rgb[color_index + 1],
            value,
        ));
    }

    colors
}

fn generate_set(
    file_name: String,
    max_iters: u32,
    colors: Vec<&str>,
    x_min: f64,
    y_min: f64,
    x_max: f64,
    y_max: f64,
    w: u32,
    h: u32,
) {
    let start = Instant::now();

    let mut buffer = RgbImage::new(w, h);
    let gradient = get_gradient(colors, max_iters);
    for x in 0..w {
        for y in 0..h {
            let x_percent = x as f64 / w as f64;
            let y_percent = y as f64 / h as f64;
            let cx = x_min + (x_max - x_min) * x_percent;
            let cy = y_min + (y_max - y_min) * y_percent;
            let iters = num_iters(cx, cy, max_iters);
            let pixel = buffer.get_pixel_mut(x, y);
            let color = gradient.get(iters as usize).unwrap_or(&[0, 0, 0]);
            *pixel = Rgb(*color);
        }
    }
    buffer.save(&file_name).unwrap();
    let duration = Instant::now() - start;
    println!("Successfully rendered frame '{}' in {:?}.", file_name, duration);
}

fn num_iters(cx: f64, cy: f64, max_iters: u32) -> u32 {
    let mut z = Complex::new(0.0, 0.0);
    let c = Complex::new(cx, cy);

    for i in 0..=max_iters {
        if z.norm() > 2.0 {
            return i;
        }
        z = z * z + c;
    }

    max_iters
}

fn main() {
    let (w, h) = (1920, 1080);
    let target_x = -0.749;
    let target_y = 0.1;
    let (a_w, a_h) = (16.0, 9.0);
    let max_iters = 5;
    let min_scale = 1;
    let max_scale = 200;

    fs::create_dir_all("./output").unwrap();

    let params: Vec<(usize, f64, f64, f64, f64)> = (min_scale..=max_scale)
        .map(|i| {
            let scale = 2.0_f64.powf(i as f64 / 10.0);
            let x_min = target_x - (a_w / scale);
            let x_max = target_x + (a_w / scale);
            let y_min = target_y - (a_h / scale);
            let y_max = target_y + (a_h / scale);
            (i, x_min, y_min, x_max, y_max)
        })
        .collect();

    params.into_par_iter().for_each(|(i, x_min, y_min, x_max, y_max)| {
        let filename = format!("output/fractal_{:0>5}.png", i);
        generate_set(
            filename,
            max_iters,
            vec!["#1C448E", "#6F8695", "#CEC288", "#FFE381", "#DBFE87"],
            x_min,
            y_min,
            x_max,
            y_max,
            w,
            h,
        );
    });
}