mod data;
mod model;
mod plot;
use data::{load_data, records_to_array, compute_rul, build_features};
use model::{train_random_forest, predict_random_forest};
use plot::{plot_cycles_vs_rul, plot_predicted_vs_true};
use ndarray::{Array2, Array1, Axis, s};
use ndarray_stats::CorrelationExt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train_data = load_data("data/PM_train.csv");
    let test_data = load_data("data/PM_test.csv");
    println!("Loaded {} training records", train_data.len());
    println!("Loaded {} testing records", test_data.len());
    let train_array = records_to_array(&train_data);
    let test_array = records_to_array(&test_data);
    let _rul_train = compute_rul(&train_data);
    let _rul_test = compute_rul(&test_data);
    let (x_train, y_train) = build_features(&train_array);
    let (x_test, y_test) = build_features(&test_array);
    println!("x_train shape: {:?}", x_train.dim());
    println!("y_train shape: {:?}", y_train.dim());
    println!("x_test shape: {:?}", x_test.dim());
    println!("y_test shape: {:?}", y_test.dim());
    let model = train_random_forest(&x_train, &y_train);
    let cycles = x_train.slice(s![.., 0]);
    plot_cycles_vs_rul(cycles, y_train.view())?;
    let y_test_pred = predict_random_forest(&model, &x_test);
    let rmse = ((&y_test - &y_test_pred).mapv(|x| x.powi(2)).mean().unwrap()).sqrt();
    println!("Test RMSE: {:.2}", rmse);
    let y_mean = y_test.mean().unwrap();
    let ss_tot = (&y_test - y_mean).mapv(|x| x.powi(2)).sum();
    let ss_res = (&y_test - &y_test_pred).mapv(|x| x.powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;
    println!("Test R² score: {:.4}", r2);
    plot_predicted_vs_true(y_test.view(), y_test_pred.view())?;
    Ok(())
}
