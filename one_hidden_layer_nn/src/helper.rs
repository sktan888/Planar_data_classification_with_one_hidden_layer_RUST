
use rand::thread_rng;
use rand::Rng;
use ndarray::{Array2,array,Array1};

pub fn generate_spiral_planar_dataset() -> (Array2<f32>, Array2<u8>) {
    /*
        Generate spiral-shaped  data 

        Arguments:
        none

        Return:
        X -- (2, 400) array 
        Y -- (1, 400) array
    */

    // generate spiral-shaped data points
    // np.random.seed(1)
    const m: usize = 400; // number of examples or points
    const N: usize = m/2; // number of points per class
    let D = 2; // number of columns in X
    let mut X: Array2<f32> = Array2::zeros((m,D)); // m rows of points and D columns of coordinates 
    let mut Y: Array2<u8> = Array2::zeros((m,1)); // labels vector (0 for red, 1 for blue)
    let a = 4; // # maximum ray of the flower, length of petal

    for j in 0..2 { // generate data for 2 classes

        // r is the radius, and t is the angle in radians
        // t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2
        // r = a*np.sin(4*t) + np.random.randn(N)*0.2

        let mut rng = thread_rng(); // creates a thread-local random number generator
        let random_numbers: Vec<f32> = (0..N)  // N random numbers collected into a vector Vec<f64>
            .map(|_| rng.gen::<f32>() * 0.2)   // scale the random number by multiplying it with 0.2
            .collect();

        // t is evenly_space_1D_array
        // let t: [f32; N] = array![j * 3.12..(j + 1) * 3.12].linspace(N).unwrap();
        // let t: [f32; N] = array![j as f32 * 3.12..(j as f32 + 1.0) * 3.12].linspace(N);

        let t: Array1<f32> = array![j as f32 * 3.12..(j as f32 + 1.0) * 3.12]
        .into_iter()
        .collect::<Vec<_>>()
        .try_into();
        match t {
            Ok(t) => {
                println!("{:?}", t);
            }
            Err(error) => {
                eprintln!("Error creating linear space: {}", error);
            }
        }
        t = t.linspace(N);

        // let r: [f32; N] = array![a*np.sin(4*j*3.12)..a*np.sin(4*(j+1)*3.12)].linspace(N).unwrap();
        let r: [f32; N] = array![a as f32 *((4.0*j as f32 *3.12).sin())..a as f32 *((4.0*(j as f32 + 1.0)*3.12).sin())].linspace(N);
        match r {
            Ok(r) => {
                println!("{:?}", r);
            }
            Err(error) => {
                eprintln!("Error creating linear space: {}", error);
            }
        }

        // iterate over each element in random numbers vector
        random_numbers.iter().enumerate().for_each(|(index, value)| {
            t[[index, 0]] += value;  // add random value to t angle
            r[[index, 0]] += value;  // add random value to t angle
        });
        
        // stacking column-wise vector of coordinates (r*np.sin(t), r*np.cos(t)) vertically into a single array, where each row rep a point
        for ix in N*j..N*(j+1) {
            let tx = t[[ix, 0]];
            let rx = r[[ix, 0]];
            X[[ix, 0]] = rx*(tx.sin());
            X[[ix, 1]] = rx*(tx.cos());
            Y[[ix,0]] = j as u8;
        }

    }
    let owned_X = X.to_owned();
    let owned_Y = Y.to_owned();   
    (owned_X.t(), owned_Y.t()) // transpose returns (D, m) and (1, m) arrays
}
