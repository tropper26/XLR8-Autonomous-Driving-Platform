use pyo3::{pyclass, pymethods};
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Eq, Hash, PartialEq, Clone, Copy)]
struct GridKey {
    x: i64,
    y: i64,
}

#[derive(PartialEq)]
struct GridValue {
    x: f64,
    y: f64,
    node_id: i64,
}

#[pyclass]
#[pyo3(frozen)]
pub struct SpatialGrid {
    min_x: f64,
    min_y: f64,
    cell_size_x: f64,
    cell_size_y: f64,
    grid: Mutex<HashMap<GridKey, Vec<GridValue>>>,
}

impl SpatialGrid {
    fn grid_key(&self, x: f64, y: f64) -> GridKey {
        let x = ((x - self.min_x) / self.cell_size_x).trunc() as i64;
        let y = ((y - self.min_y) / self.cell_size_y).trunc() as i64;
        GridKey { x, y }
    }
}

#[pymethods]
impl SpatialGrid {
    #[new]
    fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64, grid_cell_count: (f64, f64)) -> Self {
        Self {
            min_x,
            min_y,
            cell_size_x: (max_x - min_x) / grid_cell_count.0,
            cell_size_y: (max_y - min_y) / grid_cell_count.1,
            grid: Mutex::new(HashMap::new()),
        }
    }

    fn insert_node(&self, node_id: i64, x: f64, y: f64) {
        let key = self.grid_key(x, y);
        let grid_value = GridValue { x, y, node_id };

        self.grid.lock().unwrap().entry(key).or_default().push(grid_value);
    }

    fn insert_nodes(&self, nodes: Vec<(i64, f64, f64)>) {
        let mut grid = self.grid.lock().unwrap();

        for (node_id, x, y) in nodes {
            let key = self.grid_key(x, y);
            let grid_value = GridValue { x, y, node_id };

            grid.entry(key).or_default().push(grid_value);
        }
    }

    fn remove_node(&self, node_id: i64, x: f64, y: f64) {
        let key = self.grid_key(x, y);
        let mut grid = self.grid.lock().unwrap();

        if let Some(entry) = grid.get_mut(&key) {
            let to_remove = GridValue { x, y, node_id };
            entry.retain(|v| *v != to_remove);

            if entry.is_empty() {
                grid.remove(&key);
            }
        }
    }

    fn update_node(
        &self,
        node_id: i64,
        old_x: f64,
        old_y: f64,
        new_x: f64,
        new_y: f64,
    ) {
        self.remove_node(node_id, old_x, old_y);
        self.insert_node(node_id, new_x, new_y);
    }

    #[pyo3(signature = (x, y, threshold = f64::INFINITY))]
    fn get_closest_node(&self, x: f64, y: f64, threshold: f64) -> Option<(i64, f64, f64)> {
        let center_key = self.grid_key(x, y);
        let grid = self.grid.lock().unwrap();
        
        let all_keys = if threshold.is_infinite() {
            grid.keys().cloned().collect()
        } else {
            let mut keys = Vec::new();
            let threshold_x = (threshold / self.cell_size_x).trunc() as i64 + 1;
            let threshold_y = (threshold / self.cell_size_y).trunc() as i64 + 1;

            for i in -threshold_x..=threshold_x {
                for j in -threshold_y..=threshold_y {
                    keys.push(GridKey { x: center_key.x + i, y: center_key.y + j });
                }
            }

            keys
        };
        
        let mut closest_node = None;
        let mut closest_distance = threshold;

        for (grid_key, values) in grid.iter() {
            if !all_keys.contains(grid_key) {
                continue;
            }

            for value in values {
                let distance = (value.x - x).hypot(value.y - y);
                
                if distance < closest_distance {
                    closest_node = Some((value.node_id, value.x, value.y));
                    closest_distance = distance;
                }
            }
        }

        closest_node 
    }
}
