use pyo3::prelude::*;
use crate::spatial_grid::SpatialGrid;

mod spatial_grid;
mod spiral;

#[pymodule]
fn rust_optimized(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SpatialGrid>()?;
    Ok(())
}
