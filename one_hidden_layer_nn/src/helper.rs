use linfa::dataset::Records;
use linfa::prelude::Predict;
use linfa::prelude::ToConfusionMatrix;
use linfa::prelude::*;
use linfa::traits::Fit;
use linfa_logistic::LogisticRegression;
use log::info;
use ndarray::s;
use ndarray::{linspace, Array1, Array2};
use plotly::{
    color::{NamedColor, Rgb},
    common::{Marker, Mode},
    layout::{Axis, Layout, TicksDirection},
    Contour, Plot, Scatter,
};

use std::error::Error;
use std::fs::File;
use std::io::Write;

//use ndarray::arr2;
//use ndarray_npy::read_npy;
use ndarray_npy::write_npy;
//use ndarray_npy::ReadNpyExt;
//use ndarray_npy::WriteNpyExt;
//use npy::NpyData;
//use std::env;

//use ndarray::ErrorKind;
use ndarray::ShapeError;
use std::num::ParseFloatError;

use ndarray_npy::ReadNpyError;
use ndarray_npy::WriteNpyError;
//use std::convert::From;

use ndarray::Axis as OtherAxis;
use std::iter::Zip;
use crate::linear_regression::model;
use crate::linear_regression::predict;

#[derive(Debug)]
pub enum Errors {
    ShapeError(ShapeError),
    ParseFloatError(ParseFloatError),
    ReadNpyError(ReadNpyError),
    WriteNpyError(WriteNpyError),
    IoError(std::io::Error),
    MeanCalculationFailed,
}

pub struct GradientDescentResults {
    pub w: Array2<f32>,
    pub b: f32,
    pub dw: Array2<f32>,
    pub db: f32,
    pub costs: Vec<f32>,
}

pub struct ModelResults {
    pub costs: Vec<f32>,
    pub y_prediction_test: Array2<f32>,
    pub y_prediction_train: Array2<f32>,
    pub w: Array2<f32>,
    pub b: f32,
    pub learning_rate: f32,
    pub num_iterations: i32,
}

pub struct PredictionResults {
    pub y_prediction_train: Array2<f32>,
    pub y_prediction_test: Array2<f32>,
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

pub fn fit_logistic_regression_model(
    train_x: &Array2<f32>,
    train_y: &Array2<f32>,
    test_x: &Array2<f32>,
    test_y: &Array2<f32>,
) -> Result<(ModelResults), Errors> {
    //) -> Result<(ModelResults), Errors> {
    //) -> Result<(), Errors> {
    /*
    This function fits the logistic regression model according to the given training data

    Argument:
    X -- (n_features, n_samples) matrix where features refer to planar coordindates
    Y -- (n_label, n_samples) matrix labels refer to red (0.0) and blue (1.0)

    Returns:
    model
    */

    //let (_train_x, _train_y, _test_x, _test_y) = (x, y, x, y);
    let mut num_iterations = 2000;
    let mut learning_rate = 0.005;
    let print_cost = true;
    let mut costs: Vec<f32> = Vec::new(); // Create an empty vector

    let mut b = 0.0;

    let mut y_prediction_test = create_array(b)?;
    let mut y_prediction_train = create_array(b)?;
    let mut w = create_array(b)?;

    //let (_costs, _y_prediction_test, _y_prediction_train, _w, _b, _learning_rate, _num_iterations) =
    let model_result = model(
        train_x,
        train_y,
        test_x,
        test_y,
        num_iterations,
        learning_rate,
        print_cost,
    );

    // Handle model_results
    match model_result {
        Ok(model_results) => {
            // Process the successful predictions
            (
                costs,
                y_prediction_test,
                y_prediction_train,
                w,
                b,
                learning_rate,
                num_iterations,
            ) = (
                model_results.costs,
                model_results.y_prediction_test,
                model_results.y_prediction_train,
                model_results.w,
                model_results.b,
                model_results.learning_rate,
                model_results.num_iterations,
            );

            let b_array = create_array(b)?;

            // overwrite the file if it exists
            let _ = write_npy("./model/model_weights.npy", &w).map_err(|_| Errors::WriteNpyError);
            let _ =
                write_npy("./model/model_bias.npy", &b_array).map_err(|_| Errors::WriteNpyError);
            let _ = write_npy("./model/y_prediction_train.npy", &y_prediction_train)
                .map_err(|_| Errors::WriteNpyError);
            let _ = write_npy("./model/y_prediction_test.npy", &y_prediction_test)
                .map_err(|_| Errors::WriteNpyError);
        }
        Err(error) => {
            // Handle the error
            eprintln!("Error modeling: {:?}", error);
        }
    }

    Ok(ModelResults {
        costs,
        y_prediction_test,
        y_prediction_train,
        w,
        b,
        learning_rate,
        num_iterations,
    })
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
/*
pub fn plot_decision_boundary(
    x: &Array2<f32>,
    model: ModelResults,
    plot_title: &str,
    mut plot: Plot,
) {
    //pub fn plot_decision_boundary(x: &Array2<f32>, model: ModelResults, plot_title: &str) {
    /*
    This function plots the contour of decision boundary of the trained model

    Argument:
    model -- predict a 2D grid of points
    input data X -- (n_features, n_samples) scatter plot of inputs over contour
    plot_title -- name of plot

    Returns:
    a file of plot

    */

    // Set min and max values and some padding
    // Find the minimum value in each row
    let row = x.row(0).to_owned();
    let mut x_min = 0.0 as f32;
    x_min = find_minimum(&row);

    let mut x_max = 0.0 as f32;
    x_max = find_maximum(&row);

    let row = x.row(1).to_owned();
    let mut y_min = 0.0 as f32;
    y_min = find_minimum(&row);

    let mut y_max = 0.0 as f32;
    y_max = find_maximum(&row);

    let h: usize = x.shape()[1] as usize; // number of examples
    let x = linspace(x_min, x_max, h);
    let mut x: Array1<f32> = x.collect(); // (m,1)

    let y = linspace(y_min, y_max, h);
    let mut y: Array1<f32> = y.collect();

    // Generate a 2D grid of points with distance h between them
    let xx = meshgrid(&x, &y); // xx is (features,number of points) (2,m)
                               //info!("meshgrid xx shape is: {:?} ", xx.shape());

    // Predict the function value for the whole grid
    // x -- data of size (features_flatterned, number of examples)
    let predict_result = predict(&model.w, model.b, &xx);

    let mut z: Array2<f32> = Array2::zeros((1, xx.shape()[1]));
    match predict_result {
        Ok(results) => {
            // Process the successful predictions
            z = results; // <Array2<f32>
                         //info!("meshgrid prediction shape is: {:?} ", z.shape());
                         //info!("meshgrid prediction is: {:?} ", z);
        }
        Err(error) => {
            // Handle the error
            eprintln!("Error modeling: {:?}", error);
        }
    }

    /*
    // testing contour with known z array.. same output as predict z
    let mut z = Array2::zeros((1, xx.shape()[1]));
    // Efficiently set values in the second half of z to 1.0
    let mid_point = xx.shape()[1] / 2;
    z.slice_mut(s![0, mid_point..]).fill(1.0);
    // 0: This selects all rows in the first dimension (since it's a 2D array).
    // mid_point..: This selects a range of columns starting from the mid_point (inclusive) to the end of the array.
    */

    // Plot the contour and training examples
    let trace3 = Contour::new(xx.row(0).to_vec(), xx.row(1).to_vec(), z.into_raw_vec());

    // let mut plot = Plot::new(); // using given plot from earlier
    plot.add_trace(trace3);

    let layout = Layout::new()
        .title(plot_title)
        .width(500)
        .height(500)
        .x_axis(
            Axis::new()
                .title("x1")
                .grid_color(Rgb::new(211, 211, 211))
                //.range(vec![-4.0, 4.0])
                //.range(vec![-a, a])
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
                //.range(vec![-4.0, 4.0])
                //.range(vec![-a, a])
                .show_grid(true)
                .show_line(true)
                .show_tick_labels(true)
                .tick_color(Rgb::new(127, 127, 127))
                .ticks(TicksDirection::Outside)
                .zero_line(false),
        );

    plot.set_layout(layout);

    // adding text annotation

    let html = plot.to_html();

    let str1 = "./plots/";
    let str2 = ".html";
    let joined_string = format!("{}{}", str1, plot_title);
    let path = format!("{}{}", joined_string, str2);
    let mut file = File::create(path).expect("Error creating file");
    file.write_all(html.as_bytes())
        .expect("Error writing to file");
}
*/
pub fn find_minimum(row: &Array1<f32>) -> f32 {
    let mut x_min = 0.0 as f32;
    row.iter().enumerate().for_each(|(index, &value)| {
        if value < x_min {
            x_min = value;
        }
    });
    x_min -= 1.0;

    x_min
}

pub fn find_maximum(row: &Array1<f32>) -> f32 {
    let mut x_max = 0.0 as f32;
    row.iter().enumerate().for_each(|(index, &value)| {
        if value > x_max {
            x_max = value;
        }
    });
    x_max += 1.0;

    x_max
}

fn meshgrid(x: &Array1<f32>, y: &Array1<f32>) -> Array2<f32> {
    /*
    This function generates 2D grid of points

    Argument:
    x -- single array of grid value along x axis x feature
    y -- single array of grid value along y axis y feature

    Returns:
    2D array (number of features, number of points) ... (2,1000)

    */
    let (nx, ny) = (x.len(), y.len());
    let mut xx: Array2<f32> = Array2::zeros((2, nx));

    //let mut xx = Array2::zeros((ny, nx));
    //let mut yy = Array2::zeros((ny, nx));

    for (i, &x_val) in x.iter().enumerate() {
        xx[[0, i]] = x_val;
    }

    for (i, &y_val) in y.iter().enumerate() {
        xx[[1, i]] = y_val;
    }

    xx
}

/*
pub fn plot(x: &Array2<f32>, _y: &Array2<f32>, a: i32, plot_title: &str) -> Plot {
    /*
        Plot data

        Arguments:
        X -- (2, 400) array
        Y -- (1, 400) array
        a - integer length of flower petal

        Return:
        plot saved into a .html file in folder plots
    */

    let x1: Vec<f32> = x.row(0).to_vec(); // First row of X
    let x2: Vec<f32> = x.row(1).to_vec(); // Second row of X
                                          //let colors: Vec<f32> = y.row(0).to_vec(); // Values for coloring

    // the first half of x1 and x2 for first label 1 and 2 respectively
    let half_len = x1.len() / 2; // assume 2 classes of equal len
    let label1_x1: Vec<f32> = x1[0..half_len].to_vec(); // j is 0 (blue)
    let label1_x2: Vec<f32> = x2[0..half_len].to_vec();

    let label2_x1: Vec<f32> = x1[half_len..x1.len()].to_vec(); // j is 1 (red)
    let label2_x2: Vec<f32> = x2[half_len..x1.len()].to_vec();
    // Paint red for 0 and blue for 1
    let trace1 = Scatter::new(label1_x1.clone(), label1_x2.clone())
        .name("blue:0")
        .mode(Mode::Markers)
        .marker(Marker::new().color(NamedColor::Blue).size(8));

    let trace2 = Scatter::new(label2_x1.clone(), label2_x2.clone())
        .name("red:1")
        .mode(Mode::Markers)
        .marker(Marker::new().color(NamedColor::Red).size(8));

    let mut plot = Plot::new();
    plot.add_trace(trace1);
    plot.add_trace(trace2);

    let layout = Layout::new()
        .title(plot_title)
        .width(500)
        .height(500)
        .x_axis(
            Axis::new()
                .title("x1")
                .grid_color(Rgb::new(211, 211, 211))
                .range(vec![-a, a])
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
                .range(vec![-a, a])
                .show_grid(true)
                .show_line(true)
                .show_tick_labels(true)
                .tick_color(Rgb::new(127, 127, 127))
                .ticks(TicksDirection::Outside)
                .zero_line(false),
        );
    plot.set_layout(layout);

    let html = plot.to_html();

    let str1 = "./plots/";
    let str2 = ".html";
    let joined_string = format!("{}{}", str1, plot_title);
    let path = format!("{}{}", joined_string, str2);
    let mut file = File::create(path).expect("Error creating file");
    file.write_all(html.as_bytes())
        .expect("Error writing to file");

    plot
}

pub fn simple_contour_plot(plot_title: &str) {
    let n = 200;
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut z: Vec<Vec<f64>> = Vec::new();

    for index in 0..n {
        let value = -2.0 * 3.0 + 4.0 * 3.0 * (index as f64) / (n as f64);
        x.push(value);
        y.push(value);
    }

    y.iter().take(n).for_each(|y| {
        let mut row = Vec::<f64>::new();
        x.iter().take(n).for_each(|x| {
            let radius_squared = x.powf(2.0) + y.powf(2.0);
            let zv = x.sin() * y.cos() * radius_squared.sin() / (radius_squared + 1.0).log10();
            row.push(zv);
        });
        z.push(row);
    });

    let trace = Contour::new(x, y, z);
    let mut plot = Plot::new();

    plot.add_trace(trace);

    let html = plot.to_html();

    let str1 = "./plots/";
    let str2 = ".html";
    let joined_string = format!("{}{}", str1, plot_title);
    let path = format!("{}{}", joined_string, str2);
    let mut file = File::create(path).expect("Error creating file");
    file.write_all(html.as_bytes())
        .expect("Error writing to file");
}
*/