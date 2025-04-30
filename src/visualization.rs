use plotters::prelude::*;

pub fn plot_cycles_vs_rul(cycles: Vec<u32>, rul: Vec<u32>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("plots/cycles_vs_rul.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Engine Cycles vs Remaining Useful Life", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0u32..300u32, 0u32..150u32)?;
    chart.configure_mesh().draw()?;
    chart.draw_series(
        cycles.iter().zip(rul.iter()).map(|(x, y)| {
            Circle::new((*x, *y), 3, RED.filled())
        })
    )?;
    Ok(())
}
pub fn plot_predicted_vs_true(predicted: Vec<f64>, true_vals: Vec<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("plots/predicted_vs_true.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let max_val = predicted.iter().chain(true_vals.iter()).cloned().fold(0. / 0., f64::max);
    let mut chart = ChartBuilder::on(&root)
        .caption("Predicted vs True RUL", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..max_val, 0f64..max_val)?;
    chart.configure_mesh().draw()?;
    chart.draw_series(
        predicted.iter().zip(true_vals.iter()).map(|(p, t)| {
            Circle::new((*p, *t), 3, RED.filled())
        })
    )?;
    Ok(())
}