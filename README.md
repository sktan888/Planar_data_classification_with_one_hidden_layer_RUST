 [![Clippy Rust Lint with Github Actions](https://github.com/sktan888/Planar_data_classification_with_one_hidden_layer_RUST/actions/workflows/main.yml/badge.svg)](https://github.com/sktan888/Planar_data_classification_with_one_hidden_layer_RUST/actions/workflows/main.yml)

# Planar_data_classification_with_one_hidden_layer_RUST
RUST implementation of one layer neural network for planar data classification

## Set up working environment
* install RUST: ```curl https://sh.rustup.rs -sSf | sh```
* restart current shell  ``` . "$HOME/.cargo/env"  ``` in .bashrc file
* check version ``` rustc --version ```
* create Makefile for make utility : ``` touch Makefile ```
``` 
rust-version:
	rustc --version
format:
	cargo fmt --quiet
lint:
	cargo clippy --quiet
test:
	cargo test --quiet
run:
	cargo run
release:
	cargo build --release
all: format lint test run
```
* add new project ```Cargo new project_name``` 

``` tree.```

```
.
├── Cargo.lock
├── Cargo.toml
├── Makefile
├── src
│   └── main.rs
└── target
```

## Steps
    - Implement a 2-class classification neural network with a single hidden layer
    - Use the non-linear tanh activation function for hidden layer
    - Compute the cross entropy loss
    - Implement forward and backward propagation
