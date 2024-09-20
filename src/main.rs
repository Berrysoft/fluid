#![allow(clippy::reversed_empty_ranges)]

use anyhow::Result;
use image::codecs::gif::*;
use image::{Delay, Frame, RgbaImage};
use itertools::Itertools;
use ndarray::{iter::IterMut, *};
use pbr::ProgressBar;
use serde_derive::Deserialize;
use sprs::*;
use sprs_ldl::LdlNumeric;
use std::fs::{File, OpenOptions};
use std::ops::AddAssign;

struct Fluid {
    height: usize,
    width: usize,
    velosity: Array3<f64>,
    dyes: Array3<f64>,
    diffusion_solver: LdlNumeric<f64, usize>,
    pressure_solver: LdlNumeric<f64, usize>,
}

impl Fluid {
    pub fn new(height: usize, width: usize, viscosity: f64) -> Self {
        let diffusion_solver = Self::build_solver(height, width, 1., -viscosity, -1.);
        let pressure_solver = Self::build_solver(height, width, 0., 1., 1.);
        Self {
            height,
            width,
            velosity: Array::zeros((height, width, 2)),
            dyes: Array::zeros((height, width, 3)),
            diffusion_solver,
            pressure_solver,
        }
    }

    fn build_solver(
        height: usize,
        width: usize,
        r: f64,
        s: f64,
        factor: f64,
    ) -> LdlNumeric<f64, usize> {
        let cnt = height * width;
        let mut result = CsMat::new_csc((cnt, cnt), vec![0; cnt + 1], vec![], vec![]);
        for i in 0..height {
            for j in 0..width {
                let curidx = Self::coord(width, i, j);

                if i == 0 || i == height - 1 || j == 0 || j == width - 1 {
                    result.insert(curidx, curidx, 1.);
                    continue;
                }

                let mut self_total = r - 4. * s;
                for (di, dj) in [(1, 0), (0, -1), (-1, 0), (0, 1)] {
                    let ni = i as isize + di;
                    let nj = j as isize + dj;

                    if ni == 0 || ni == height as isize - 1 || nj == 0 || nj == width as isize - 1 {
                        self_total += factor * s;
                    } else {
                        result.insert(curidx, Self::coord(width, ni as usize, nj as usize), s);
                    }
                }
                result.insert(curidx, curidx, self_total);
            }
        }
        LdlNumeric::new(result.view()).unwrap()
    }

    const fn coord(width: usize, i: usize, j: usize) -> usize {
        i * width + j
    }

    pub fn splat(&mut self, center: &[usize; 2], r: usize, accel: &[f64; 2], dyes: &[f64; 3]) {
        let r = r as isize;
        for dx in -r..r + 1 {
            for dy in -r..r + 1 {
                let x = center[0] as isize + dx;
                let y = center[1] as isize + dy;

                if x < 0 || x >= self.height as isize || y < 0 || y >= self.width as isize {
                    continue;
                }

                let x = x as usize;
                let y = y as usize;

                let dr = ((dx * dx + dy * dy) as f64).sqrt();

                if dr <= r as f64 {
                    let mut v = self.velosity.slice_mut(s![x, y, ..]);
                    v += &ArrayView::from(&accel);

                    let mut dye = self.dyes.slice_mut(s![x, y, ..]);
                    dye += &ArrayView::from(dyes);
                    clip_iter(dye.iter_mut(), None, Some(1.0));
                }
            }
        }
    }

    pub fn step(&mut self) {
        let mut op = self.velosity.clone();
        let mut di = op.slice_mut(s![.., .., 0]);
        di *= -1.;
        di += &Array1::from_iter((0..self.height).map(|a| a as f64)).slice(s![.., NewAxis]);
        let mut dj = op.slice_mut(s![.., .., 1]);
        dj *= -1.;
        dj += &Array1::from_iter((0..self.width).map(|a| a as f64)).slice(s![NewAxis, ..]);

        let mut advected_velocity = self.bilinear(&self.velosity, &op);
        let mut advected_dyes = self.bilinear(&self.dyes, &op);

        self.copy_to_boundary3(&mut advected_dyes, 0.);
        self.dyes = advected_dyes;
        self.copy_to_boundary3(&mut advected_velocity, -1.);

        let mut diffused_velocity = Array3::zeros((self.height, self.width, 2));
        for dim in 0..2 {
            diffused_velocity.slice_mut(s![.., .., dim]).assign(
                &self
                    .diffusion_solver
                    .solve(&Array1::from_iter(
                        advected_velocity.slice(s![.., .., dim]).map(|a| *a),
                    ))
                    .to_shape((self.height, self.width))
                    .unwrap(),
            );
        }
        self.copy_to_boundary3(&mut diffused_velocity, -1.);

        let div = self.diverg(&diffused_velocity);
        let mut q = self
            .pressure_solver
            .solve(&Array1::from_iter(div))
            .into_shape_with_order((self.height, self.width))
            .unwrap();
        self.copy_to_boundary2(&mut q, 1.);
        let g = self.grad(&q);
        diffused_velocity -= &g;
        self.copy_to_boundary3(&mut diffused_velocity, -1.);
        self.velosity = diffused_velocity;
    }

    fn grad(&self, f: &Array2<f64>) -> Array3<f64> {
        let mut result = Array3::zeros((self.height, self.width, 2));
        result
            .slice_mut(s![1..-1, .., 0])
            .assign(&((&f.slice(s![2.., ..]) - &f.slice(s![..-2, ..])) / 2.));
        result
            .slice_mut(s![.., 1..-1, 1])
            .assign(&((&f.slice(s![.., 2..]) - &f.slice(s![.., ..-2])) / 2.));
        result
    }

    fn diverg(&self, f: &Array3<f64>) -> Array2<f64> {
        let mut result = Array2::zeros((self.height, self.width));
        let f_x = f.slice(s![.., .., 0]);
        result
            .slice_mut(s![1..-1, ..])
            .assign(&((&f_x.slice(s![2.., ..]) - &f_x.slice(s![..-2, ..])) / 2.));
        let f_y = f.slice(s![.., .., 1]);
        result
            .slice_mut(s![.., 1..-1])
            .add_assign(&((&f_y.slice(s![.., 2..]) - &f_y.slice(s![.., ..-2])) / 2.));
        result
    }

    fn read_field(&self, f: &Array3<f64>, i: &Array2<usize>, j: &Array2<usize>) -> Array3<f64> {
        let shape = i.shape();
        let mut result = Array3::zeros((shape[0], shape[1], f.shape()[2]));
        for x in 0..shape[0] {
            for y in 0..shape[1] {
                let rx = clip(i[[x, y]], 0, self.height - 1);
                let ry = clip(j[[x, y]], 0, self.width - 1);
                result
                    .slice_mut(s![x, y, ..])
                    .assign(&f.slice(s![rx, ry, ..]));
            }
        }
        result
    }

    fn bilinear(&self, f: &Array3<f64>, op: &Array3<f64>) -> Array3<f64> {
        let x = op.slice(s![.., .., 0]);
        let (xr, xf) = Self::floor(x);
        let mut dx = x.to_owned();
        dx -= &xf;

        let y = op.slice(s![.., .., 1]);
        let (yr, yf) = Self::floor(y);
        let mut dy = y.to_owned();
        dy -= &yf;

        let tl = self.read_field(f, &xr, &yr);
        let tr = self.read_field(f, &xr, &(yr.clone() + 1));
        let bl = self.read_field(f, &(xr.clone() + 1), &yr);
        let br = self.read_field(f, &(xr + 1), &(yr + 1));

        let dx = dx.slice_move(s![.., .., NewAxis]);
        let dy = dy.slice_move(s![.., .., NewAxis]);
        let l = tl * (&dx * -1. + 1.) + bl * &dx;
        let r = tr * (&dx * -1. + 1.) + br * &dx;

        l * (&dy * -1. + 1.) + r * dy
    }

    fn floor(a: ArrayView2<f64>) -> (Array2<usize>, Array2<f64>) {
        let shape = a.shape();
        let mut ru = Array2::zeros((shape[0], shape[1]));
        let mut rf = Array2::zeros((shape[0], shape[1]));
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                rf[[i, j]] = a[[i, j]].floor();
                ru[[i, j]] = rf[[i, j]] as usize;
            }
        }
        (ru, rf)
    }

    fn copy_to_boundary2(&self, f: &mut Array2<f64>, factor: f64) {
        {
            let (mut v1, v2) = f.view_mut().split_at(Axis(0), 1);
            v1.assign(&v2.slice(s![0, ..]));
            v1 *= factor;
        }
        {
            let (v2, mut v1) = f.view_mut().split_at(Axis(0), self.height - 1);
            v1.assign(&v2.slice(s![-1, ..]));
            v1 *= factor;
        }
        {
            let (mut v1, v2) = f.view_mut().split_at(Axis(1), 1);
            let (v3, _) = v2.split_at(Axis(1), 1);
            v1.assign(&v3);
            v1 *= factor;
        }
        {
            let (v2, mut v1) = f.view_mut().split_at(Axis(1), self.width - 1);
            let (_, v3) = v2.split_at(Axis(1), self.width - 2);
            v1.assign(&v3);
            v1 *= factor;
        }
    }

    fn copy_to_boundary3(&self, f: &mut Array3<f64>, factor: f64) {
        {
            let (mut v1, v2) = f.view_mut().split_at(Axis(0), 1);
            v1.assign(&v2.slice(s![0, .., ..]));
            v1 *= factor;
        }
        {
            let (v2, mut v1) = f.view_mut().split_at(Axis(0), self.height - 1);
            v1.assign(&v2.slice(s![-1, .., ..]));
            v1 *= factor;
        }
        {
            let (mut v1, v2) = f.view_mut().split_at(Axis(1), 1);
            let (v3, _) = v2.split_at(Axis(1), 1);
            v1.assign(&v3);
            v1 *= factor;
        }
        {
            let (v2, mut v1) = f.view_mut().split_at(Axis(1), self.width - 1);
            let (_, v3) = v2.split_at(Axis(1), self.width - 2);
            v1.assign(&v3);
            v1 *= factor;
        }
    }
}

fn clip_iter<D: Dimension>(v: IterMut<f64, D>, min: Option<f64>, max: Option<f64>) {
    for item in v {
        if let Some(min) = min {
            if *item < min {
                *item = min;
            }
        }
        if let Some(max) = max {
            if *item > max {
                *item = max;
            }
        }
    }
}

fn clip(v: usize, min: usize, max: usize) -> usize {
    if v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

#[derive(Debug, Deserialize)]
struct Input {
    width: usize,
    height: usize,
    time: usize,
    viscosity: f64,
    splats: Vec<Splat>,
}

#[derive(Debug, Deserialize)]
struct Splat {
    time_from: usize,
    time_to: usize,
    center: [usize; 2],
    radius: usize,
    accel: [f64; 2],
    dyes: [f64; 3],
}

fn main() -> Result<()> {
    let input = File::open("input.json")?;
    let input: Input = serde_json::from_reader(input)?;
    let mut sim = Fluid::new(input.height, input.width, input.viscosity);

    let mut frames = vec![];
    let mut pb = ProgressBar::new(input.time as u64);
    pb.set_max_refresh_rate(Some(std::time::Duration::from_millis(100)));

    for step in 0..input.time {
        pb.inc();
        for splat in &input.splats {
            if splat.time_from <= step && splat.time_to > step {
                sim.splat(&splat.center, splat.radius, &splat.accel, &splat.dyes);
            }
        }

        sim.step();

        let frame = Frame::from_parts(
            RgbaImage::from_raw(
                input.width as u32,
                input.height as u32,
                sim.dyes
                    .iter()
                    .chunks(3)
                    .into_iter()
                    .flat_map(|mut chunk| {
                        let r = chunk.next().unwrap();
                        let g = chunk.next().unwrap();
                        let b = chunk.next().unwrap();
                        [(r * 255.) as u8, (g * 255.) as u8, (b * 255.) as u8, 255u8]
                    })
                    .collect(),
            )
            .unwrap(),
            0,
            0,
            Delay::from_saturating_duration(std::time::Duration::from_millis(30)),
        );
        frames.push(frame);
    }
    pb.finish();

    let gif_output = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("output.gif")?;

    let mut encoder = GifEncoder::new_with_speed(gif_output, 10);
    encoder.set_repeat(Repeat::Infinite)?;
    encoder.encode_frames(frames)?;

    let f = hdf5::File::create("output.hdf5")?;
    f.new_dataset_builder()
        .with_data(sim.velosity.view())
        .create("velocity")?;
    f.new_dataset_builder()
        .with_data(sim.dyes.view())
        .create("dyes")?;
    Ok(())
}
