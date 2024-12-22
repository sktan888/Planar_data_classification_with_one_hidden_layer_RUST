use one_hidden_layer_nn::data::injest;

#[test]
fn test_injest() {
    let (x, y) = injest();
    let shape_x = x.shape();
    let shape_y = y.shape();
    let m = shape_y[1];

    let expected = 400;
    let result = m;
    assert_eq!(result, expected, "test_injest algo failed");
}
