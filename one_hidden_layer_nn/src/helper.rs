use log::info;
//use linfa::prelude::*;
use linfa::dataset::Records;
use linfa::prelude::Predict;
use linfa::prelude::ToConfusionMatrix;
use linfa::traits::Fit;
use linfa_logistic::LogisticRegression;

use ndarray::{linspace, Array1, Array2};
use plotly::{
    color::{NamedColor, Rgb},
    common::{Marker, Mode},
    layout::{Axis, Layout, TicksDirection},
    Contour, Plot, Scatter,
};
use rand::thread_rng;
use rand::Rng;
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
    w: Array2<f32>,
    b: f32,
    dw: Array2<f32>,
    db: f32,
    costs: Vec<f32>,
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
    y_prediction_train: Array2<f32>,
    y_prediction_test: Array2<f32>,
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

pub fn sigmoid(z: Array2<f32>) -> Array2<f32> {
    /*
    Compute the sigmoid of z as 1 / (1 + np.exp(-z))
    Apply the exponential function to each element

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z), a probability between 0 and 1
    */

    1.0 / (1.0 + z.mapv(|x| (-x).exp()))
}

// try Result error next time
pub fn initialize_with_zeros(dim: usize) -> (Array2<f32>, f32) {
    //pub fn initialize_with_zeros(dim: usize) -> Result<(Array2<f32>, f32), String> {
    /*
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias) of type float
    */

    let w: Array2<f32> = Array2::zeros((dim, 1));
    let owned_w = w.to_owned();
    let b = 0.0;
    (owned_w, b)
}

// use dot matrix slower than python
pub fn propagate(
    w: &Array2<f32>,
    b: f32,
    x: &Array2<f32>,
    y: &Array2<f32>,
) -> (Array2<f32>, f32, f32) {
    /*
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    grads -- dictionary containing the gradients of the weights and bias
            (dw -- gradient of the loss with respect to w, thus same shape as w)
            (db -- gradient of the loss with respect to b, thus same shape as b)
    cost -- negative log-likelihood cost for logistic regression

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    */

    // m = X.shape[1] // (num_px * num_px * 3, number of examples)
    let m: f32 = x.shape()[1] as f32; // cast a `usize` to an `f32`

    /*
    # FORWARD PROPAGATION (FROM X TO COST)
    # compute activation
    # compute cost by using np.dot to perform multiplication.
    # And don't use loops for the sum.
    */

    let a = sigmoid(w.t().dot(x) + b);

    let cost: f32 =
        -(y * (&a.mapv(|e| e.ln())) + (1.0 - y) * ((1.0 - &a).mapv(|d| d.ln()))).sum() / m;

    //# BACKWARD PROPAGATION (TO FIND GRAD)

    let dw = x.dot(&((&a - y).t())) / m; // // Negate each element

    let db: f32 = (&a - y).sum() / m;

    (dw, db, cost)
}

pub fn optimize(
    w: &Array2<f32>,
    b: f32,
    x: &Array2<f32>,
    y: &Array2<f32>,
    num_iterations: i32,
    learning_rate: f32,
    print_cost: bool,
) -> Result<GradientDescentResults, Box<dyn std::error::Error>> {
    // (Array2<f32>, f32, Array2<f32>, f32, Vec<f32>)
    /*
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    */

    let mut w_owned = w.to_owned();
    let mut b_owned = b;

    let mut costs = Vec::new(); // Create an empty vector

    let mut dw: Array2<f32> = Array2::zeros((x.shape()[0], 1)); // (row (features), col (examples)) refers to (num_px * num_px * 3, number of examples)
    let mut db = 0.0;
    let mut cost = 0.0;

    for i in 0..num_iterations {
        (dw, db, cost) = propagate(&w_owned, b_owned, x, y);

        w_owned = w_owned - learning_rate * &dw; // Dereference w_owned and apply element-wise multiplication
        b_owned -= learning_rate * db;

        // Record the costs print interval
        let print_interval = 100;
        if i % print_interval == 0 {
            costs.push(cost);

            if print_cost {
                println!("Cost after iteration {:?}: {:?}", i, cost);
                info!("Cost after iteration {:?}: {:?}", i, cost);
            }
        }
    }

    Ok(GradientDescentResults {
        w: w_owned,
        b: b_owned,
        dw,
        db,
        costs,
    })
}

pub fn predict(w: &Array2<f32>, b: f32, x: &Array2<f32>) -> Result<Array2<f32>, Errors> {
    /*
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    */

    let m = x.shape()[1];

    let mut y_prediction: Array2<f32> = Array2::zeros((1, m));

    assert_eq!(w.shape(), &[x.shape()[0], 1]);

    let _ = w.to_shape((x.shape()[0], 1)).map_err(Errors::ShapeError);

    let a = sigmoid((w.t()).dot(x) + b);

    //# Using no loop for better efficieny
    //# Y_prediction[A > 0.5] = 1

    // Iterate over the elements of 'a' and assign values to 'y_prediction'
    for ((i, j), value) in a.indexed_iter() {
        if *value > 0.5 {
            y_prediction[(i, j)] = 1.0;
        }
    }

    Ok(y_prediction)
}

pub fn model(
    x_train: &Array2<f32>,
    y_train: &Array2<f32>,
    x_test: &Array2<f32>,
    y_test: &Array2<f32>,
    num_iterations: i32,
    learning_rate: f32,
    print_cost: bool,
) -> Result<ModelResults, Errors> {
    /*
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to True to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    */

    /*
    # initialize parameters with zeros
    # w, b = ...

    # Gradient descent
    # params, grads, costs = ...

    # Retrieve parameters w and b from dictionary "params"
    # w = ...
    # b = ...

    # Predict test/train set examples
    # Y_prediction_test = ...
    # Y_prediction_train = ...
    */

    let mut prediction_data = PredictionResults {
        y_prediction_train: y_train.clone(),
        y_prediction_test: y_test.clone(),
    };

    let (w, b) = initialize_with_zeros(x_train.shape()[0]);

    let Ok(results) = optimize(
        &w,
        b,
        x_train,
        y_train,
        num_iterations,
        learning_rate,
        print_cost,
    ) else {
        todo!()
    };

    let (w, b, _dw, _db, costs) = (results.w, results.b, results.dw, results.db, results.costs);

    let y_prediction_test_result = predict(&w, b, x_test);
    let y_prediction_train_result = predict(&w, b, x_train);

    // Handle y_prediction_test_result
    match y_prediction_test_result {
        Ok(predictions) => {
            // Process the successful predictions
            prediction_data.y_prediction_test = predictions;
        }
        Err(error) => {
            // Handle the error
            eprintln!("Error predicting test data: {:?}", error);
        }
    }

    // Handle y_prediction_train_result
    match y_prediction_train_result {
        Ok(predictions) => {
            // Process the successful predictions
            prediction_data.y_prediction_train = predictions;
        }
        Err(error) => {
            // Handle the error
            eprintln!("Error predicting test data: {:?}", error);
        }
    }

    if print_cost {
        // Refactor with References: If you only need to read the data from prediction_data.y_prediction_train, consider using a reference (&) to avoid unnecessary moves:

        match (&prediction_data.y_prediction_train - y_train).abs().mean() {
            Some(mean) => {
                println!("train accuracy: {:.2}", 100.0 * (1.0 - mean));
                info!("train accuracy: {:.2}", 100.0 * (1.0 - mean));
            }
            None => {
                // Handle the case where the mean is None
                // (e.g., return an error, use a default value)
                // (0.0) // Or another appropriate default value
            }
        }

        // Refactor with References: If you only need to read the data from prediction_data.y_prediction_train, consider using a reference (&) to avoid unnecessary moves:
        match (&prediction_data.y_prediction_test - y_test).abs().mean() {
            Some(mean) => {
                println!("test accuracy: {:.2}", 100.0 * (1.0 - mean));
                info!("test accuracy: {:.2}", 100.0 * (1.0 - mean));
            }
            None => {
                // Handle the case where the mean is None
                // (e.g., return an error, use a default value)
                // (0.0) // Or another appropriate default value
            }
        }
    }

    Ok(ModelResults {
        costs,
        y_prediction_test: prediction_data.y_prediction_test,
        y_prediction_train: prediction_data.y_prediction_train,
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

pub fn plot_decision_boundary(x: &Array2<f32>, model: ModelResults, plot_title: &str) {
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
    info!("meshgrid xx shape is: {:?} ", xx.shape());

    // Predict the function value for the whole grid
    // x -- data of size (features_flatterned, number of examples)
    let predict_result = predict(&model.w, model.b, &xx);

    let mut z: Array2<f32> = Array2::zeros((1, xx.shape()[1]));
    match predict_result {
        Ok(results) => {
            // Process the successful predictions
            z = results; // <Array2<f32>
            info!("meshgrid prediction shape is: {:?} ", z.shape());
            info!("meshgrid prediction is: {:?} ", z);
        }
        Err(error) => {
            // Handle the error
            eprintln!("Error modeling: {:?}", error);
        }
    }

    // Plot the contour and training examples
    let trace3 = Contour::new(xx.row(0).to_vec(), xx.row(1).to_vec(), z.into_raw_vec());

    let mut plot = Plot::new();
    plot.add_trace(trace3);
    let html = plot.to_html();

    let str1 = "./plots/";
    let str2 = ".html";
    let joined_string = format!("{}{}", str1, plot_title);
    let path = format!("{}{}", joined_string, str2);
    let mut file = File::create(path).expect("Error creating file");
    file.write_all(html.as_bytes())
        .expect("Error writing to file");
}

fn find_minimum(row: &Array1<f32>) -> f32 {
    let mut x_min = 0.0 as f32;
    row.iter().enumerate().for_each(|(index, &value)| {
        if value < x_min {
            x_min = value;
        }
    });
    x_min -= 1.0;

    x_min
}

fn find_maximum(row: &Array1<f32>) -> f32 {
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

pub fn plot(x: &Array2<f32>, _y: &Array2<f32>, a: i32, plot_title: &str) {
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
        .title(plot_title)
        //.title("Flower Data")
        .width(500)
        .height(500)
        .x_axis(
            Axis::new()
                .title("x1")
                .grid_color(Rgb::new(211, 211, 211))
                //.range(vec![-4.0, 4.0])
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
                //.range(vec![-4.0, 4.0])
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

pub fn generate_flower_planar_dataset(m: usize, a: i32) -> (Array2<f32>, Array2<f32>) {
    /*
        Generate spiral-shaped  data

        Arguments:
        m: number of points or examples
        a:amplitude or length of petal

        Return:
        X -- (2, 400) array
        Y -- (1, 400) array
    */

    // generate spiral-shaped data points
    // const M: usize = 400; // number of examples or points
    // const N: usize = M / 2; // number of points per class
    let n = m / 2; // 2 classes of equal number of examples
    let d = 2; // assuming 2 = number of columns in X
    let mut x: Array2<f32> = Array2::zeros((m, d)); // m rows of points and D columns of coordinates
    let mut y: Array2<f32> = Array2::zeros((m, 1)); // labels vector (0.0 for red, 1.0 for blue)
                                                    // let a = 4; // # maximum ray of the flower, length of petal

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
        //r.iter().enumerate().for_each(|(_index, value)| {
        r.iter().for_each(|value| {
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
        let mut tx: f32 = 0.0;
        let mut rx: f32 = 0.0;

        for ix in n * j..n * (j + 1) {
            if ix < n {
                tx = t[ix];
                rx = r[ix];
            } else {
                tx = t[ix - n];
                rx = r[ix - n];
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
