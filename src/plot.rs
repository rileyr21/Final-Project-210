use ndarray::{ArrayView1};
use plotters::prelude::*;
use std::error::Error;

pub fn plot_cycles_vs_rul(cycles: ArrayView1<f64>, rul: ArrayView1<f64>) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("plots/cycles_vs_rul.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Cycles vs. RUL", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..cycles.fold(0.0, |a, &b| a.max(b)), 0f64..rul.fold(0.0, |a, &b| a.max(b)))?;
    chart.configure_mesh().draw()?;
    chart.draw_series(
        cycles.iter().zip(rul.iter()).map(|(&x, &y)| Circle::new((x, y), 2, RED.filled())),
    )?;
    Ok(())
}

pub fn plot_predicted_vs_true(y_true: ArrayView1<f64>, y_pred: ArrayView1<f64>) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("plots/predicted_vs_true.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Predicted vs True RUL", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..y_true.fold(0.0, |a, &b| a.max(b)), 0f64..y_true.fold(0.0, |a, &b| a.max(b)))?;
    chart.configure_mesh().draw()?;
    chart.draw_series(
        y_true.iter().zip(y_pred.iter()).map(|(&t, &p)| Circle::new((t, p), 2, BLUE.filled())),
    )?;
    Ok(())
}