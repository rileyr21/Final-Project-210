use ndarray::{Array1, Array2, Axis, array, s};
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug)]
pub struct EngineRecord {
    pub id: u32,
    pub cycle: u32,
    pub setting1: f64,
    pub setting2: f64,
    pub setting3: f64,
}

pub fn load_data(filename: &str) -> Vec<EngineRecord> {
    let file = File::open(filename).expect("Failed to open file");
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for line in reader.lines() {
        if let Ok(line) = line {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 8 {
                let record = EngineRecord {
                    id: parts[0].parse().unwrap_or(0),
                    cycle: parts[1].parse().unwrap_or(0),
                    setting1: parts[2].parse().unwrap_or(0.0),
                    setting2: parts[3].parse().unwrap_or(0.0),
                    setting3: parts[4].parse().unwrap_or(0.0),
                };
                records.push(record);
            }
        }
    }
    records
}

pub fn records_to_array(records: &[EngineRecord]) -> Array2<f64> {
    let data: Vec<f64> = records.iter()
        .flat_map(|record| {
            vec![
                record.id as f64,
                record.cycle as f64,
                record.setting1,
                record.setting2,
                record.setting3,
            ]
        })
        .collect();

    Array2::from_shape_vec((records.len(), 5), data)
        .expect("Failed to build Array2 from records")
}

pub fn compute_rul(records: &[EngineRecord]) -> Array1<f64> {
    use std::collections::HashMap;
    let mut max_cycles: HashMap<u32, u32> = HashMap::new();
    for record in records {
        max_cycles
            .entry(record.id)
            .and_modify(|e| *e = (*e).max(record.cycle))
            .or_insert(record.cycle);
    }
    Array1::from(
        records
            .iter()
            .map(|r| {
                let max_cycle = max_cycles.get(&r.id).copied().unwrap_or(r.cycle);
                if r.cycle > max_cycle {
                    0.0
                } else {
                    (max_cycle - r.cycle) as f64
                }
            })
            .collect::<Vec<_>>(),
    )
}

pub fn build_features(x: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let cycles = x.slice(s![.., 1]);
    let max_cycle = cycles.fold(0.0_f64, |acc, &v| acc.max(v));
    let normalized_cycles = cycles.mapv(|c| c / max_cycle);
    let normalized_cycles = normalized_cycles.insert_axis(Axis(1));
    let other_features = x.slice(s![.., 2..]);
    let features = ndarray::concatenate(Axis(1), &[normalized_cycles.view(), other_features.view()])
        .expect("Failed to concatenate features");
    let rul = x.column(1).to_owned();
    (features, rul)
}