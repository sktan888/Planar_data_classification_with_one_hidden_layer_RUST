// use fern::Dispatch;
use log::info;
use log::LevelFilter;
use one_hidden_layer_nn::data::injest;
use one_hidden_layer_nn::helper::fit_logistic_regression_model;
use one_hidden_layer_nn::helper::plot;

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

    /*
    let prediction_data = PredictionResults { 
        y_prediction_train: Array2::zeros((m_train, 1)).t(), 
        y_prediction_test: Array2::zeros((m_test, 1)).t(),
    }; 
    */
    println!("The shape of x is: {:?}", shape_x);
    println!("The shape of y is: {:?}", shape_y);
    println!("There are m = {:?} testing examples ", m_examples);

    info!("The shape of x is: {:?}", shape_x);
    info!("The shape of y is: {:?}", shape_y);
    info!("There are m = {:?} testing examples ", m_examples);

    // Visualise datasets
    let plot_title="train dataset";
    plot(&train_x, &train_y, a, plot_title);

    let plot_title="test dataset";
    plot(&test_x, &test_y, a, plot_title);

    //let _ = linfa_logistic_regression();
    let _ = fit_logistic_regression_model(&train_x, &train_y, &test_x, &test_y);
    //plot_decision_boundary(clf, X, Y);
    //compute_accuracy(clf, X, Y);

    Ok(())
}
