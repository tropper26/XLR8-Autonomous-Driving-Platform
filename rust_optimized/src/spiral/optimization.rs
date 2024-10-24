use crate::spiral::auto_generated_math::{bending_energy, compute_cf_gradient, soft_constraints};
use pyo3::prelude::{PyAnyMethods, PyModule};
use pyo3::types::PyTuple;
use pyo3::{pyclass, pyfunction, pymethods, Bound};

const ALPHA: f64 = 10000.0;
const BETA: f64 = 10000.0;
const GAMMA: f64 = 10000.0;

pub type SpiralParameters = [f64; 5];

#[pyclass]
#[pyo3(frozen)]
pub struct OptimizeFunction {
    p_0: f64,
    p_3: f64,
    x_0: f64,
    y_0: f64,
    psi_0: f64,
    x_f: f64,
    y_f: f64,
    psi_f: f64,
}

#[pymethods]
impl OptimizeFunction {
    #[allow(unused_variables)]
    #[pyo3(signature = (x, *args))]
    fn __call__(&self, x: [f64; 3], args: &Bound<'_, PyTuple>) -> f64 {
        let p = [self.p_0, x[0], x[1], self.p_3, x[2]];

        compute_cost_function(
            &p, ALPHA, BETA, GAMMA, self.x_0, self.y_0, self.psi_0, self.x_f, self.y_f, self.psi_f,
        )
    }
}

#[pyclass]
#[pyo3(frozen)]
pub struct JacobianFunction {
    p_0: f64,
    p_3: f64,
    x_0: f64,
    y_0: f64,
    psi_0: f64,
    x_f: f64,
    y_f: f64,
    psi_f: f64,
}

#[pymethods]
impl JacobianFunction {
    #[allow(unused_variables)]
    #[pyo3(signature = (x, *args))]
    fn __call__(&self, x: Vec<f64>, args: &Bound<'_, PyTuple>) -> [f64; 3] {
        let p = [self.p_0, x[0], x[1], self.p_3, x[2]];

        compute_cf_gradient(
            &p, ALPHA, BETA, GAMMA, self.x_0, self.y_0, self.psi_0, self.x_f, self.y_f, self.psi_f,
        )
    }
}

#[pyfunction]
#[pyo3(signature = (scipy_optimize_module, x_0, y_0, psi_0, k_0, x_f, y_f, psi_f, k_f, k_max, equal = false))]
pub fn optimize_spiral(
    scipy_optimize_module: &Bound<PyModule>,
    x_0: f64,
    y_0: f64,
    psi_0: f64,
    k_0: f64,
    x_f: f64,
    y_f: f64,
    psi_f: f64,
    k_f: f64,
    k_max: f64,
    equal: bool,
) -> SpiralParameters {
    let p_0 = k_0;
    let p_3 = k_f;

    let arc_length_initial_guess = (x_f - x_0).hypot(y_f - y_0);
    let init_param = vec![0.0, 0.0, arc_length_initial_guess];

    let k_min = if equal { k_max } else { -k_max };
    let bounds = vec![
        (Some(k_min), Some(k_max)),
        (Some(k_min), Some(k_max)),
        (
            Some(arc_length_initial_guess),
            Some(3.5 * arc_length_initial_guess),
        ),
    ];

    let objective_function = OptimizeFunction {
        p_0,
        p_3,
        x_0,
        y_0,
        psi_0,
        x_f,
        y_f,
        psi_f,
    };
    let jac = JacobianFunction {
        p_0,
        p_3,
        x_0,
        y_0,
        psi_0,
        x_f,
        y_f,
        psi_f,
    };

    let result = scipy_optimize_module
        .call_method1(
            "minimize",
            (
                objective_function,
                init_param,
                (),
                "L-BFGS-B",
                jac,
                (),
                (),
                bounds,
            ),
        )
        .expect("Failed to call minimize");

    let optimized_params: Vec<f64> = result
        .getattr("x")
        .expect("Failed to get x")
        .extract()
        .expect("Failed to extract optimized parameters");

    [
        p_0,
        optimized_params[0],
        optimized_params[1],
        p_3,
        optimized_params[2],
    ]
}

fn compute_cost_function(
    p: &SpiralParameters,
    alpha: f64,
    beta: f64,
    gamma: f64,
    x_0: f64,
    y_0: f64,
    psi_0: f64,
    x_f: f64,
    y_f: f64,
    psi_f: f64,
) -> f64 {
    bending_energy(p) + soft_constraints(p, alpha, beta, gamma, x_0, y_0, psi_0, x_f, y_f, psi_f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn math_test() {
        let x_0 = 4.53;
        let y_0 = -17.54;
        let psi_0 = 0.45;
        let x_f = -3.45;
        let y_f = 534.54;
        let psi_f = 2.55;
        let alpha = 0.1;
        let beta = 0.2;
        let gamma = 0.3;
        let k_0 = 0.5;
        let k_f = 0.6;

        let x_init = [-0.75, 0.12, 34.0];
        let p = [k_0, x_init[0], x_init[1], k_f, x_init[2]];

        let cost_value =
            compute_cost_function(&p, alpha, beta, gamma, x_0, y_0, psi_0, x_f, y_f, psi_f);
        let jac = compute_cf_gradient(&p, alpha, beta, gamma, x_0, y_0, psi_0, x_f, y_f, psi_f);

        fn assert_float(a: f64, b: f64) {
            assert!((a - b).abs() < f64::EPSILON);
        }

        assert_float(cost_value, 61171.778972973254);
        assert_float(jac[0], -4901.41452708606);
        assert_float(jac[1], -3195.792650968944);
        assert_float(jac[2], 28.059179938084558);
    }
}
