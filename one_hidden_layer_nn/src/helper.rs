use ndarray::{array, linspace, Array1, Array2};
use rand::thread_rng;
use rand::Rng;
use log::info;
use std::fs::File;
use std::io::Write;

use plotly::{
    color::{NamedColor, Rgb, Rgba, Color},
    common::{
        ColorScale, ColorScalePalette, DashType, Fill, Font, Line, LineShape, Marker, Mode,
        Orientation, Pattern, PatternShape,
    },
    layout::{Axis, BarMode, CategoryOrder, Layout, Legend, TicksDirection, TraceOrder},
    sankey::{Line as SankeyLine, Link, Node},
    traces::table::{Cells, Header},
    Bar, Plot, Sankey, Scatter, ScatterPolar, Table, ImageFormat,
};

pub fn plot(x: &Array2<f32>, y: &Array2<f32>){
    let x1: Vec<f32> = x.row(0).to_vec(); // First row of X
    let x2: Vec<f32> = x.row(1).to_vec();  // Second row of X
    let colors: Vec<f32> = y.row(0).to_vec();  // Values for coloring

    // the first half of x1 and x2 for first label 1 and 2 respectively
    let half_len = x1.len() / 2; 
    let label1_x1: Vec<f32> = x1[0..half_len].to_vec(); 
    let label1_x2: Vec<f32> = x2[0..half_len].to_vec(); 

    let label2_x1: Vec<f32> = x1[half_len..x1.len()].to_vec(); 
    let label2_x2: Vec<f32> = x2[half_len..x1.len()].to_vec(); 
    // Paint red for 0 and blue for 1
    let trace1 = Scatter::new(label1_x1.clone(), label1_x2.clone())
    .name("red:0")
    .mode(Mode::Markers)
    .marker(Marker::new()
    .color(NamedColor::Red).size(8)); 

    let trace2 = Scatter::new(label2_x1.clone(), label2_x2.clone())
    .name("blue:1")
    .mode(Mode::Markers)
    .marker(Marker::new()
    .color(NamedColor::Blue).size(8)); 

    let mut plot = Plot::new();
    plot.add_trace(trace1);
    plot.add_trace(trace2);

    let layout = Layout::new()
    .title("Flower Data")
    .width(500)
    .height(500)
    .x_axis(
        Axis::new()
        .title("x1")
        .grid_color(Rgb::new(211, 211, 211))
        .range(vec![-4.0, 4.0])
        .show_grid(true)
        .show_line(true)
        .show_tick_labels(true)
        .tick_color(Rgb::new(127, 127, 127))
        .ticks(TicksDirection::Outside)
        .zero_line(false),
    )
    .y_axis(
        Axis::new()
        .title("x2")
        .grid_color(Rgb::new(211, 211, 211))
        .range(vec![-4.0, 4.0])
        .show_grid(true)
        .show_line(true)
        .show_tick_labels(true)
        .tick_color(Rgb::new(127, 127, 127))
        .ticks(TicksDirection::Outside)
        .zero_line(false),
    );
    //.x_axis(Axis::new().title("x1").range(vec![-4.0, 4.0]))
    //.y_axis(Axis::new().title("x2").range(vec![-4.0, 4.0]));
    plot.set_layout(layout);


    let html = plot.to_html();
    let mut file = File::create("./plots/flower_dataset.html").expect("Error creating file");
    file.write_all(html.as_bytes()).expect("Error writing to file");

}

pub fn generate_spiral_planar_dataset() -> (Array2<f32>, Array2<f32>) {
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
    let mut y: Array2<f32> = Array2::zeros((m, 1)); // labels vector (0.0 for red, 1.0 for blue)
    let a = 4; // # maximum ray of the flower, length of petal

    for j in 0..2 {
        // generate data for 2 classes

        // r is the radius, and t is the angle in radians

        // np.random.randn(N)*0.2
        let mut rng = thread_rng(); // creates a thread-local random number generator
        let random_numbers: Vec<f32> = (0..n) // N random numbers collected into a vector Vec<f64>
            .map(|_| rng.gen::<f32>() * 0.2) // scale the random number by multiplying it with 0.2
            .collect();
        // info!("random_numbers for t is: {:?}", random_numbers);

        // t = np.linspace(j*3.12,(j+1)*3.12,N)  
        let start = j as f32 * 3.12;
        let end = (j as f32 + 1.0) * 3.12;
        let t = linspace(start, end, n);
        let mut t: Vec<f32> = t.collect();

        // iterate t vector and add random values
        // t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2
        
        random_numbers
            .iter()
            .enumerate()
            .for_each(|(index, value)| {
                t[index] += *value; // add random value to t for theta angle
            });
            

        // r = a*np.sin(4*t)
        let mut r: Vec<f32> = t.clone();
        // let mut rr: f32=0.0;

        let mut modified_r = Vec::with_capacity(r.len());
        r.iter().enumerate().for_each(|(index, value)| {
            let rr = *value * 4.0;
            modified_r.push((a as f32) * rr.sin());
        });
        r = modified_r;
        // info!("r is: {:?}", r);

        // iterate r vector and add random values
        // r = a*np.sin(4*t) + np.random.randn(N)*0.2

        let random_numbers: Vec<f32> = (0..n) // N random numbers collected into a vector Vec<f64>
        .map(|_| rng.gen::<f32>() * 0.2) // scale the random number by multiplying it with 0.2
        .collect();

        //info!("random_numbers for r is: {:?}", random_numbers);

        random_numbers
            .iter()
            .enumerate()
            .for_each(|(index, value)| {
                r[index] += *value; 
            });
        // info!("r is: {:?}", r);
        
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
            y[[ix, 0]] = j as f32;
        }
    }

    let xt = x.t().to_owned();
    let yt = y.t().to_owned();

    //info!("xt is: {:?}", xt);
    // info!("yt is: {:?}", yt);
    (xt, yt)
}
