use ndarray::{array, linspace, Array1, Array2};
use rand::thread_rng;
use rand::Rng;
use log::info;
use plotly::{Plot, Scatter, ImageFormat};
use plotly::common::ColorScale;
use std::fs::File;
use std::io::Write;


pub fn plot(x: &Array2<f32>, y: &Array2<u8>){
    let x1: Vec<f32> = x.row(0).to_vec(); // First row of X
    let x2: Vec<f32> = x.row(1).to_vec();  // Second row of X
    let colors: Vec<u8> = y.row(0).to_vec();  // Values for coloring

    // Normalize colors to a range of 0.0 to 1.0
    let normalized_colors: Vec<f64> = colors.iter().map(|&c| (c as f64)).collect();
    /*
    let trace = Scatter::new(x1.clone(), x2.clone())
        .mode(plotly::common::Mode::Markers)
        .marker(plotly::common::Marker::new()
            .size(10)
            .color(normalized_colors)
            .color_scale(plotly::common::ColorScale::Viridis));
    */
    let trace = Scatter::new(x1.clone(), x2.clone())
                .name("trace");

    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.show(); // Show the plot in a browser window
    plot.write_image("./out.png", ImageFormat::PNG, 1024, 680, 1.0);
    // Plotly is a helpful tool when building interact web applications in Rust
}

pub fn generate_spiral_planar_dataset() -> (Array2<f32>, Array2<u8>) {
    /*
        Generate spiral-shaped  data

        Arguments:
        none

        Return:
        X -- (2, 400) array
        Y -- (1, 400) array
    */

    // generate spiral-shaped data points
    const m: usize = 400; // number of examples or points
    const n: usize = m / 2; // number of points per class
    let d = 2; // number of columns in X
    let mut x: Array2<f32> = Array2::zeros((m, d)); // m rows of points and D columns of coordinates
    let mut y: Array2<u8> = Array2::zeros((m, 1)); // labels vector (0 for red, 1 for blue)
    let a = 4; // # maximum ray of the flower, length of petal

    for j in 0..2 {
        // generate data for 2 classes

        // r is the radius, and t is the angle in radians
        // t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2
        // r = a*np.sin(4*t) + np.random.randn(N)*0.2

        let mut rng = thread_rng(); // creates a thread-local random number generator
        let random_numbers: Vec<f32> = (0..n) // N random numbers collected into a vector Vec<f64>
            .map(|_| rng.gen::<f32>() * 0.2) // scale the random number by multiplying it with 0.2
            .collect();

        let start = j as f32 * 3.12;
        let end = (j as f32 + 1.0) * 3.12;
        let t = linspace(start, end, n);
        let mut t: Vec<f32> = t.collect();

        let start = a as f32 * ((4.0 * j as f32 * 3.12).sin());
        let end = a as f32 * ((4.0 * (j as f32 + 1.0) * 3.12).sin());
        let r = linspace(start, end, n);
        let mut r: Vec<f32> = r.collect();

        // iterate over each element in random numbers vector
        random_numbers
            .iter()
            .enumerate()
            .for_each(|(index, value)| {
                t[index] += *value; // add random value to t for theta angle
                r[index] += *value; // add random value to r for radii
            });

        // stacking column-wise vector of coordinates (r*np.sin(t), r*np.cos(t)) vertically into a single array, where each row rep a point
        let mut tx: f32=0.0;
        let mut rx: f32=0.0;

        for ix in n * j..n * (j + 1) {
            
            if ix < n {
                tx = t[ix];
                rx = r[ix];
            } else{
                tx = t[ix-n];
                rx = r[ix-n];
            }
            x[[ix, 0]] = rx * (tx.sin());
            x[[ix, 1]] = rx * (tx.cos());
            y[[ix, 0]] = j as u8;
        }
    }

    let xt = x.t().to_owned();
    let yt = y.t().to_owned();
    (xt, yt)
}
