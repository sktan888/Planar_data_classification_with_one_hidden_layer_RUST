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

use crate::helper::find_maximum;
use crate::helper::find_minimum;
use crate::helper::meshgrid;
use crate::linear_regression::predict;

use crate::helper::GradientDescentResults;
use crate::helper::ModelResults;
use crate::helper::PredictionResults;

pub fn plot_costs(c: Vec<f32>, plot_title: &str) -> Plot {
    let y_data: Vec<f32> = c.clone();

    let n = y_data.len(); // Number of elements in the vector
    let mut x_data: Vec<f32> = Vec::with_capacity(n);
    for i in 1..=n {
        x_data.push(i as f32);
    }

    let trace = Scatter::new(x_data, y_data)
        .name("cost")
        .mode(Mode::Markers)
        .marker(Marker::new().color(NamedColor::Green).size(8));

    let mut plot = Plot::new();

    plot.add_trace(trace);

    let layout = Layout::new()
        .title(plot_title)
        .width(500)
        .height(500)
        .x_axis(
            Axis::new()
                .title("iterations")
                .grid_color(Rgb::new(211, 211, 211))
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
                .title("cost")
                .grid_color(Rgb::new(211, 211, 211))
                //.range(vec![-a, a])
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
    let mut x_min = 0.0_f32;
    x_min = find_minimum(&row);

    let mut x_max = 0.0_f32;
    x_max = find_maximum(&row);

    let row = x.row(1).to_owned();
    let mut y_min = 0.0_f32;
    y_min = find_minimum(&row);

    let mut y_max = 0.0_f32;
    y_max = find_maximum(&row);

    let h: usize = x.shape()[1]; // number of examples
    let x = linspace(x_min, x_max, h);
    let x: Array1<f32> = x.collect(); // (m,1)

    let y = linspace(y_min, y_max, h);
    let y: Array1<f32> = y.collect();

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
    // let trace3 = Contour::new(xx.row(0).to_vec(), xx.row(1).to_vec(), z.into_raw_vec());

    let (z_ptr, _z_offset) = z.into_raw_vec_and_offset();
    let trace3 = Contour::new(xx.row(0).to_vec(), xx.row(1).to_vec(), z_ptr);

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

    let html = plot.to_html();
    let str1 = "./plots/";
    let str2 = ".html";
    let joined_string = format!("{}{}", str1, plot_title);
    let path = format!("{}{}", joined_string, str2);
    let mut file = File::create(path).expect("Error creating file");
    file.write_all(html.as_bytes())
        .expect("Error writing to file");
}
