use smartcore::ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters};
use ndarray::{Array1, Array2};

pub fn train_random_forest(
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
) -> RandomForestRegressor<f64, f64, Array2<f64>, Array1<f64>> {
    let params = RandomForestRegressorParameters::default();
    RandomForestRegressor::fit(x_train, y_train, params)
        .expect("Failed to train Random Forest Regressor")
}

pub fn predict_random_forest(
    model: &RandomForestRegressor<f64, f64, Array2<f64>, Array1<f64>>,
    x_test: &Array2<f64>,
) -> Array1<f64> {
    model.predict(x_test)
        .expect("Failed to predict with Random Forest")
}