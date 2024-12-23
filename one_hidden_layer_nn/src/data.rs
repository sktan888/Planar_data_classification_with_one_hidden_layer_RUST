use crate::helper::generate_flower_planar_dataset;
use log::info;
use ndarray::Array2;

pub fn injest(m: i32, a: i32) -> (Array2<f32>, Array2<f32>) {
    /*
        Load dataset

        Arguments:
        none

        Return:
        X -- (2, 400) array
        Y -- (1, 400) array
    */

    let (x, y) = generate_flower_planar_dataset(m, a); // Loading data

    let shape_x = x.shape();
    let shape_y = y.shape();
    let m_examples = shape_y[1];

    println!("The shape of x is: {:?}", shape_x);
    println!("The shape of y is: {:?}", shape_y);
    println!("There are m = {:?} training examples ", m_examples);

    info!("The shape of x is: {:?}", shape_x);
    info!("The shape of y is: {:?}", shape_y);
    info!("There are m = {:?} training examples ", m_examples);

    (x, y)
}
