use log::info;
//use linfa::prelude::*;
use linfa::dataset::Records;
use linfa::prelude::Predict;
use linfa::prelude::ToConfusionMatrix;
use linfa::traits::Fit;
use linfa_logistic::LogisticRegression;

use ndarray::{linspace, Array2};
use plotly::{
    color::{NamedColor, Rgb},
    common::{Marker, Mode},
    layout::{Axis, Layout, TicksDirection},
    Plot, Scatter,
};
use rand::thread_rng;
use rand::Rng;
use std::error::Error;
use std::fs::File;
use std::io::Write;

use ndarray::{arr2, Array2};
use ndarray_npy::read_npy;
use ndarray_npy::write_npy;
use ndarray_npy::ReadNpyExt;
use ndarray_npy::WriteNpyExt;
use npy::NpyData;
use std::env;

use ndarray::ErrorKind;
use ndarray::ShapeError;
use std::error::Error;
use std::num::ParseFloatError;

use ndarray_npy::ReadNpyError;
use ndarray_npy::WriteNpyError;
use std::convert::From;


#[derive(Debug)]
enum Errors {
    ShapeError(ShapeError),
    ParseFloatError(ParseFloatError),
    ReadNpyError(ReadNpyError),
    WriteNpyError(WriteNpyError),
    ImageError(ImageError),
    IoError(std::io::Error),
}

/*
use plotly::{
    color::{Color, NamedColor, Rgb, Rgba},
    common::{
        ColorScale, ColorScalePalette, DashType, Fill, Font, Line, LineShape, Marker, Mode,
        Orientation, Pattern, PatternShape,
    },
    layout::{Axis, BarMode, CategoryOrder, Layout, Legend, TicksDirection, TraceOrder},
    sankey::{Line as SankeyLine, Link, Node},
    traces::table::{Cells, Header},
    Bar, ImageFormat, Plot, Sankey, Scatter, ScatterPolar, Table,
};
*/

fn create_array(b: f32) -> Result<Array2<f32>, Errors> {
    /* let result = Array2::from_shape_vec((1, 1), vec![b]).map_err(Errors::ShapeError);
    result */
    Array2::from_shape_vec((1, 1), vec![b]).map_err(Errors::ShapeError)
}

fn fit_logistic_regression_model(x: &Array2<f32>,y: &Array2<f32>,) -> Result<(), Errors> {
    /*
    This function fits the logistic regression model according to the given training data

    Argument:
    X -- (n_features, n_samples) matrix (2, 400) representing 400 points, 2 (x1, x2) coordindates
    Y -- (n_label, n_samples) matrix (1, 400) representing label: red (0.0) and blue (1.0)

    Returns:
    model 
    */

    let (_train_x, _train_y, _test_x, _test_y) = (x, y, x, y);
    let _num_iterations = 2000;
    let _learning_rate = 0.005;
    let print_cost = true;
    let _costs: Vec<f32> = Vec::new(); // Create an empty vector

    let _b = 0.0;

    let _y_prediction_test = create_array(_b)?;
    let _y_prediction_train = create_array(_b)?;
    let _w = create_array(_b)?;

    let (costs, _y_prediction_test, _y_prediction_train, _w, _b, _learning_rate, _num_iterations) =
        model(
            &_train_x,
            &_train_y,
            &_test_x,
            &_test_y,
            _num_iterations,
            _learning_rate,
            print_cost,
        );

    let b_array = create_array(_b)?;

    // overwrite the file if it exists
    let _ = write_npy("model_weights.npy", &_w).map_err(|_| Errors::WriteNpyError);
    let _ = write_npy("model_bias.npy", &b_array).map_err(|_| Errors::WriteNpyError);
    let _ = write_npy("test_set_x.npy", &_test_x).map_err(|_| Errors::WriteNpyError);
    let _ = write_npy("test_set_y.npy", &_y_prediction_test).map_err(|_| Errors::WriteNpyError);

    //info!("main model_cmd: b {:?}.", b_array);
    // info!("main model_cmd: w {:?}.", _w);
    // info!("main model_cmd w shape {:?}", _w.shape());
    //info!("main model_cmd: cost {:?}.", costs);

    Ok(())
}

pub fn linfa_logistic_regression() -> Result<(), Box<dyn Error>> {
    // everything above 6.5 is considered a good wine
    let (train, valid) = linfa_datasets::winequality()
        .map_targets(|x| if *x > 6 { "good" } else { "bad" })
        .split_with_ratio(0.9);

    println!(
        "Fit Logistic Regression classifier with #{} training points",
        train.nsamples()
    );
    info!("train.nsamples: {:?}", train);

    // fit a Logistic regression model with 150 max iterations
    let model = LogisticRegression::default()
        .max_iterations(150)
        .fit(&train)
        .unwrap();

    // predict and map targets
    let pred = model.predict(&valid);

    // create a confusion matrix
    let cm = pred.confusion_matrix(&valid).unwrap();

    // Print the confusion matrix, this will print a table with four entries. On the diagonal are
    // the number of true-positive and true-negative predictions, off the diagonal are
    // false-positive and false-negative
    println!("{:?}", cm);

    // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // predicted and targets)
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    let shape_train = train.records.shape();

    info!("The shape of train is: {:?}", shape_train);

    Ok(())
}

pub fn plot(x: &Array2<f32>, _y: &Array2<f32>) {
    /*
        Plot data

        Arguments:
        X -- (2, 400) array
        Y -- (1, 400) array

        Return:
        plot saved into a .html file in folder plots
    */

    let x1: Vec<f32> = x.row(0).to_vec(); // First row of X
    let x2: Vec<f32> = x.row(1).to_vec(); // Second row of X
                                          //let colors: Vec<f32> = y.row(0).to_vec(); // Values for coloring

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
        .marker(Marker::new().color(NamedColor::Red).size(8));

    let trace2 = Scatter::new(label2_x1.clone(), label2_x2.clone())
        .name("blue:1")
        .mode(Mode::Markers)
        .marker(Marker::new().color(NamedColor::Blue).size(8));

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
    plot.set_layout(layout);

    let html = plot.to_html();
    let mut file = File::create("./plots/flower_dataset.html").expect("Error creating file");
    file.write_all(html.as_bytes())
        .expect("Error writing to file");
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
    const M: usize = 400; // number of examples or points
    const N: usize = M / 2; // number of points per class
    let d = 2; // number of columns in X
    let mut x: Array2<f32> = Array2::zeros((M, d)); // m rows of points and D columns of coordinates
    let mut y: Array2<f32> = Array2::zeros((M, 1)); // labels vector (0.0 for red, 1.0 for blue)
    let a = 4; // # maximum ray of the flower, length of petal

    for j in 0..2 {
        // generate data for 2 classes

        // r is the radius, and t is the angle in radians

        // np.random.randn(N)*0.2
        let mut rng = thread_rng(); // creates a thread-local random number generator
        let random_numbers: Vec<f32> = (0..N) // N random numbers collected into a vector Vec<f64>
            .map(|_| rng.gen::<f32>() * 0.2) // scale the random number by multiplying it with 0.2
            .collect();
        // info!("random_numbers for t is: {:?}", random_numbers);

        // t = np.linspace(j*3.12,(j+1)*3.12,N)
        let start = j as f32 * 3.12;
        let end = (j as f32 + 1.0) * 3.12;
        let t = linspace(start, end, N);
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
        //r.iter().enumerate().for_each(|(_index, value)| {
        r.iter().for_each(|value| {
            let rr = *value * 4.0;
            modified_r.push((a as f32) * rr.sin());
        });
        r = modified_r;
        // info!("r is: {:?}", r);

        // iterate r vector and add random values
        // r = a*np.sin(4*t) + np.random.randn(N)*0.2

        let random_numbers: Vec<f32> = (0..N) // N random numbers collected into a vector Vec<f64>
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
        let mut tx: f32 = 0.0;
        let mut rx: f32 = 0.0;

        for ix in N * j..N * (j + 1) {
            if ix < N {
                tx = t[ix];
                rx = r[ix];
            } else {
                tx = t[ix - N];
                rx = r[ix - N];
            }
            x[[ix, 0]] = rx * (tx.sin());
            x[[ix, 1]] = rx * (tx.cos());
            y[[ix, 0]] = j as f32;
        }
    }

    let xt = x.t().to_owned();
    let yt = y.t().to_owned();

    //info!("xt is: {:?}", xt);
    //info!("yt is: {:?}", yt);
    (xt, yt)
}
