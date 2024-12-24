// use fern::Dispatch;
use log::info;
use log::LevelFilter;
use ndarray::Array2;
use one_hidden_layer_nn::data::injest;
use one_hidden_layer_nn::helper::fit_logistic_regression_model;
use one_hidden_layer_nn::helper::plot;
use one_hidden_layer_nn::helper::plot_decision_boundary;
use one_hidden_layer_nn::helper::ModelResults;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the logger
    let _ = fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{} [{}] [{}] {}",
                chrono::Local::now().format("%d-%m-%Y %H:%M:%S"),
                record.level(),
                record.target(),
                message
            ))
        })
        .level(LevelFilter::Debug)
        .chain(fern::log_file("./log/info.log")?)
        .apply();

    // Prepare datasets
    let a = 10; // maximum ray of the flower, length of petal
    let m_train = 1000; // number of examples or points of train dataset
    let (train_x, train_y) = injest(m_train, a);

    let shape_x = train_x.shape();
    let shape_y = train_y.shape();
    let m_examples = shape_y[1];

    println!("The shape of x is: {:?}", shape_x);
    println!("The shape of y is: {:?}", shape_y);
    println!("There are m = {:?} training examples ", m_examples);

    info!("The shape of x is: {:?}", shape_x);
    info!("The shape of y is: {:?}", shape_y);
    info!("There are m = {:?} training examples ", m_examples);

    let m_test = 100; // number of examples or points of test dataset
    let (test_x, test_y) = injest(m_test, a);

    let shape_x = test_x.shape();
    let shape_y = test_y.shape();
    let m_examples = shape_y[1];

    println!("The shape of x is: {:?}", shape_x);
    println!("The shape of y is: {:?}", shape_y);
    println!("There are m = {:?} testing examples ", m_examples);

    info!("The shape of x is: {:?}", shape_x);
    info!("The shape of y is: {:?}", shape_y);
    info!("There are m = {:?} testing examples ", m_examples);

    // Visualise datasets
    let plot_title = "train dataset";
    plot(&train_x, &train_y, a, plot_title);

    let plot_title = "test dataset";
    plot(&test_x, &test_y, a, plot_title);

    //let _ = linfa_logistic_regression();

    let mut modelLR = ModelResults {
        costs: Vec::new(),
        y_prediction_test: Array2::zeros((1, m_test)),
        y_prediction_train: Array2::zeros((1, m_train)),
        w: Array2::zeros((shape_x[0], 1)),
        b: 0.0,
        learning_rate: 0.0,
        num_iterations: 0,
    };

    let model_result = fit_logistic_regression_model(&train_x, &train_y, &test_x, &test_y);

    match model_result {
        Ok(model_results) => {
            // Process the successful predictions
            modelLR = model_results;
        }
        Err(error) => {
            // Handle the error
            eprintln!("Error modeling: {:?}", error);
        }
    }
    
    info!("main train_x shape is: {:?} ", train_x.shape() );
    info!("main model.w shape is: {:?} ", modelLR.w.shape() );
    info!("main model.b is: {:?} ", modelLR.b );

    let plot_title = "decision boundary";
    plot_decision_boundary(&train_x, modelLR, plot_title);

    //compute_accuracy(modelLR, train_y, test_y);

    Ok(())
}
