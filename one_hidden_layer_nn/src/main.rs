use log::LevelFilter;
use log::{debug, error, info};
use fern::Dispatch;
use use crate::data::injest;

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

     let (_X, _Y) = injest();
     
     Ok(())
}
