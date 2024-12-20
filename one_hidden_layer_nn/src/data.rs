use crate::helper::generate_spiral_planar_dataset;
use log::info;
use ndarray::{Array2};

pub fn injest() -> (Array2<f32>, Array2<u8>) {
    /*
        Load dataset

        Arguments:
        none

        Return:
        X -- (2, 400) array 
        Y -- (1, 400) array
    */

    let (X, Y) = generate_spiral_planar_dataset(); // Loading data

    let shape_X = X.shape();
    let shape_Y = Y.shape();
    let m = shape_Y[1];

    println!("The shape of X is: {:?}", shape_X);
    println!("The shape of Y is: {:?}", shape_Y);
    println!("There are m = {:?} training examples ", m);

    info!("The shape of X is: {:?}", shape_X);
    info!("The shape of Y is: {:?}", shape_Y);
    info!("There are m = {:?} training examples ", m);

    /*
        # Visualize the data:
        plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.colormaps.get_cmap("viridis"))
        plt.savefig('myPlots/load_planar_dataset.png')
    */

    (
        X, 
        Y 
    )
}