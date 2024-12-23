// use fern::Dispatch;
// use log::info;
use log::LevelFilter;
use one_hidden_layer_nn::data::injest;
//use one_hidden_layer_nn::helper::linfa_logistic_regression;
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
        .chain(fern::log_file("info.log")?)
        .apply();

    let m = 400; // number of examples or points
    let a = 4; // maximum ray of the flower, length of petal
    let (x, y) = injest(m, a);
    plot(&x, &y);
    //let _ = linfa_logistic_regression();
    //clf = fit_logistic_regression_model(X, Y);
    //plot_decision_boundary(clf, X, Y);
    //compute_accuracy(clf, X, Y);

    Ok(())
}
