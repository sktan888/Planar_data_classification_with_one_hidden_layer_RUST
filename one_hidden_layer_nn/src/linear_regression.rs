use log::info;
use ndarray::s;
use ndarray::{linspace, Array1, Array2};
use crate::helper::GradientDescentResults;
use crate::helper::Errors;
use crate::helper::ModelResults;
use crate::helper::PredictionResults;

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