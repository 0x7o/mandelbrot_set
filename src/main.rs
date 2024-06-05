use image::{Rgb, RgbImage};
use libc::{c_double, c_int, c_uint};
use rayon::prelude::*;
use std::fs;
use std::time::Instant;

extern "C" {
    fn calculate_mandelbrot(
        cx: *mut c_double,
        cy: *mut c_double,
        num_points: c_int,
        max_iters: c_uint,
        output: *mut c_uint,
    );
}

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
    resolution: u32,
) {
    let start = Instant::now();

    let mut buffer = RgbImage::new(resolution, resolution);
    let gradient = get_gradient(colors, max_iters);
    let mut h_cx = vec![];
    let mut h_cy = vec![];
    let mut output = vec![0; (resolution * resolution) as usize];

    for x in 0..resolution {
        for y in 0..resolution {
            let x_percent = x as f64 / resolution as f64;
            let y_percent = y as f64 / resolution as f64;
            let cx = x_min + (x_max - x_min) * x_percent;
            let cy = y_min + (y_max - y_min) * y_percent;
            h_cx.push(cx);
            h_cy.push(cy);
        }
    }

    unsafe {
        calculate_mandelbrot(
            h_cx.as_mut_ptr(),
            h_cy.as_mut_ptr(),
            (resolution * resolution) as c_int,
            max_iters,
            output.as_mut_ptr(),
        );
    }

    for (x, row) in output.chunks(resolution as usize).enumerate() {
        for (y, column) in row.iter().enumerate() {
            let pixel = buffer.get_pixel_mut(x as u32, y as u32);
            let color = gradient.get(*column as usize).unwrap_or(&[0, 0, 0]);
            *pixel = Rgb(*color);
        }
    }

    buffer.save(&file_name).unwrap();
    let duration = Instant::now() - start;
    println!("Rendered frame '{}' in {:?}.", file_name, duration);
}

fn main() {
    let resolution = 2048;
    let target_x = -0.7499662068297848;
    let target_y = 0.04280600288545834;
    let max_iters = 1000;
    let max_scale = 10_i64.pow(15);
    let fps = 60;
    let seconds = 30;
    let frames = fps * seconds;

    fs::create_dir_all("./output").unwrap();

    let params: Vec<(usize, f64, f64, f64, f64)> = (0..frames)
        .map(|i| {
            let scale = 10.0_f64.powf((i as f64 / frames as f64) * max_scale.ilog10() as f64);
            let x_min = target_x - (1.0 / scale);
            let x_max = target_x + (1.0 / scale);
            let y_min = target_y - (1.0 / scale);
            let y_max = target_y + (1.0 / scale);
            (i, x_min, y_min, x_max, y_max)
        })
        .collect();

    params
        .into_par_iter()
        .for_each(|(i, x_min, y_min, x_max, y_max)| {
            let filename = format!("output/fractal_{:0>5}.png", i);
            generate_set(
                filename,
                max_iters,
                vec!["#0063C7", "#B0C400", "#C40C00"],
                x_min,
                y_min,
                x_max,
                y_max,
                resolution
            );
        });
}
