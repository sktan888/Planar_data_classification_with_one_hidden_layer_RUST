use crate::helper::generate_flower_planar_dataset;
use log::info;
use ndarray::Array2;

pub fn injest(m: usize, a: i32) -> (Array2<f32>, Array2<f32>) {
    /*
        Load dataset

        Arguments:
        none

        Return:
        X -- (2, 400) array
        Y -- (1, 400) array
    */

    let (x, y) = generate_flower_planar_dataset(m, a); // Loading data

    (x, y)
}
