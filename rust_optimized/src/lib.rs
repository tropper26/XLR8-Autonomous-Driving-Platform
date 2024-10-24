use pyo3::prelude::*;
use crate::spatial_grid::SpatialGrid;
use crate::spiral::optimization::{JacobianFunction, OptimizeFunction};

mod spatial_grid;
mod spiral;

#[pyclass(eq)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct WaypointWithHeading {
    #[pyo3(get, set)]
    pub x: f64,
    #[pyo3(get, set)]
    pub y: f64,
    #[pyo3(get, set)]
    pub heading: f64,
    #[pyo3(get, set)]
    pub index_in_route: Option<i64>,
}

#[pymethods]
impl WaypointWithHeading {
    #[new]
    #[pyo3(signature = (x, y, heading, index_in_route=None))]
    pub fn new(x: f64, y: f64, heading: f64, index_in_route: Option<i64>) -> Self {
        WaypointWithHeading {
            x,
            y,
            heading,
            index_in_route,
        }
    }
}

#[pymodule]
fn rust_optimized(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WaypointWithHeading>()?;

    m.add_class::<OptimizeFunction>()?;
    m.add_class::<JacobianFunction>()?;
    m.add_function(wrap_pyfunction!(spiral::eval_spiral, m)?)?;
    m.add_function(wrap_pyfunction!(spiral::optimization::optimize_spiral, m)?)?;

    m.add_class::<SpatialGrid>()?;
    Ok(())
}
