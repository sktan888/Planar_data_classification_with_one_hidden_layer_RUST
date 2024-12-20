use crate::helper::generate_spiral_planar_dataset;
use log::info;
use ndarray::Array2;

pub fn injest() -> (Array2<f32>, Array2<u8>) {
    /*
        Load dataset

        Arguments:
        none

        Return:
        X -- (2, 400) array
        Y -- (1, 400) array
    */

    let (x, y) = generate_spiral_planar_dataset(); // Loading data

    let shape_x = x.shape();
    let shape_y = y.shape();
    let m = shape_y[1];

    println!("The shape of x is: {:?}", shape_x);
    println!("The shape of y is: {:?}", shape_y);
    println!("There are m = {:?} training examples ", m);

    info!("The shape of x is: {:?}", shape_x);
    info!("The shape of y is: {:?}", shape_y);
    info!("There are m = {:?} training examples ", m);

    /*
        # Visualize the data:
        plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.colormaps.get_cmap("viridis"))
        plt.savefig('myPlots/load_planar_dataset.png')
    */

    (x, y)
}
