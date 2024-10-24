use crate::spiral::optimization::SpiralParameters;
use ndarray::Array1;
use pyo3::prelude::*;

pub fn abcds_f(p: &SpiralParameters) -> [f64; 5] {
    let a = p[0];
    let b = -(11.0 * p[0] / 2.0 - 9.0 * p[1] + 9.0 * p[2] / 2.0 - p[3]) / p[4];
    let c = (9.0 * p[0] - 45.0 * p[1] / 2.0 + 18.0 * p[2] - 9.0 * p[3] / 2.0) / p[4].powi(2);
    let d = -(9.0 * p[0] / 2.0 - 27.0 * p[1] / 2.0 + 27.0 * p[2] / 2.0 - 9.0 * p[3] / 2.0)
        / p[4].powi(3);
    let arc_length = p[4];
    [a, b, c, d, arc_length]
}

fn cumulative_trapezoid(
    y: &[f64],
    x: &[f64],
    sum_offset: f64,
    first_element_is_zero: bool,
) -> Vec<f64> {
    assert_eq!(y.len(), x.len(), "y and x must have the same length");

    let mut result = Vec::with_capacity(y.len());

    if first_element_is_zero {
        result.push(sum_offset);
    }

    let mut cumulative_sum = sum_offset;

    for i in 1..y.len() {
        let dx_i = x[i] - x[i - 1];
        let dy_i = y[i] + y[i - 1];
        cumulative_sum += (dx_i * dy_i) / 2.0;
        result.push(cumulative_sum);
    }

    result
}

#[pyfunction]
pub fn eval_spiral(p: SpiralParameters, x_0: f64, y_0: f64, psi_0: f64, ds: f64) -> [Vec<f64>; 5] {
    let [a, b, c, d, arc_length] = abcds_f(&p);

    let point_count = (arc_length / ds).ceil() as usize;
    let s_values = Array1::linspace(0.0, arc_length, point_count);
    let curvature_values = s_values.powi(3) * d + s_values.powi(2) * c + &s_values * b + a;
    let psi_values = d * s_values.powi(4) / 4.0
        + c * s_values.powi(3) / 3.0
        + b * s_values.powi(2) / 2.0
        + a * &s_values
        + psi_0;

    let s_values = s_values.to_vec();
    let x_values = cumulative_trapezoid(&psi_values.cos().to_vec(), &s_values, x_0, true);
    let y_values = cumulative_trapezoid(&psi_values.sin().to_vec(), &s_values, y_0, true);

    [
        s_values,
        x_values,
        y_values,
        psi_values.to_vec(),
        curvature_values.to_vec(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absds_f_test() {
        let p = [0.5, -0.75, 0.12, 0.6, 34.0];
        let [a, b, c, d, arc_length] = abcds_f(&p);

        assert_eq!(a, 0.5);
        assert_eq!(b, -0.2776470588235294);
        assert_eq!(c, 0.018023356401384083);
        assert_eq!(d, -0.0002873753307551394);
        assert_eq!(arc_length, 34.0);
    }

    #[test]
    fn test_eval_spiral() {
        let p = [0.5, -0.75, 0.12, 0.6, 34.0];
        let x_0 = 4.53;
        let y_0 = -17.54;
        let psi_0 = 0.45;
        let ds = 0.01;

        let [s_values, y_values, psi_values, x_values, curvature_values] =
            eval_spiral(p, x_0, y_0, psi_0, ds);

        let expected_size = 3400;
        assert_eq!(expected_size, s_values.len());
        assert_eq!(expected_size, y_values.len());
        assert_eq!(expected_size, psi_values.len());
        assert_eq!(expected_size, x_values.len());
        assert_eq!(expected_size, curvature_values.len());
    }
}
