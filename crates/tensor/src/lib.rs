//! CPU tensor storage and automatic-differentiation operations.
use std::cell::RefCell;
use std::rc::Rc;

struct TensorData {
    data: Vec<f64>,
    shape: Vec<usize>,
    grad: Vec<f64>,
    children: Vec<Tensor>,
    backward: Box<dyn Fn()>,
}

#[derive(Clone)]
struct Tensor(Rc<RefCell<TensorData>>);

impl TensorData {
    fn new(data: Vec<f64>, shape: Vec<usize>) -> TensorData {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "data length {} doesn't match shape {:?} (which needs {} elements)",
            data.len(),
            shape,
            expected
        );
        TensorData {
            data,
            shape,
            grad: vec![0.0; expected],
            children: Vec::new(),
            backward: Box::new(|| {}),
        }
    }
}

impl Tensor {
    fn new(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor(Rc::new(RefCell::new(TensorData::new(data, shape))))
    }

    fn from_op(data: Vec<f64>, shape: Vec<usize>, children: Vec<Tensor>) -> Tensor {
        let tensor = Tensor::new(data, shape);
        tensor.0.borrow_mut().children = children;
        tensor
    }

    fn add(&self, other: &Tensor) -> Tensor {
        let self_shape = self.0.borrow().shape.clone();
        let other_shape = other.0.borrow().shape.clone();

        let same_shape = self_shape == other_shape;
        let broadcasts_right_row =
            self_shape.len() == 2 && other_shape.len() == 1 && self_shape[1] == other_shape[0];

        assert!(
            same_shape || broadcasts_right_row,
            "add: shape mismatch: left {:?}, right {:?}",
            self_shape,
            other_shape
        );

        let self_data = self.0.borrow().data.clone();
        let other_data = other.0.borrow().data.clone();

        let data: Vec<f64> = if same_shape {
            self_data
                .iter()
                .zip(other_data.iter())
                .map(|(a, b)| a + b)
                .collect()
        } else {
            let columns = self_shape[1];

            // Row-major storage repeats the raw vector for every matrix row
            // Modulo maps each flat matrix position back to its column.
            self_data
                .iter()
                .enumerate()
                .map(|(index, value)| value + other_data[index % columns])
                .collect()
        };
        let broadcast_columns = if broadcasts_right_row {
            Some(other_shape[0])
        } else {
            None
        };

        let out = Tensor::from_op(data, self_shape, vec![self.clone(), other.clone()]);

        let self_clone = self.clone();
        let other_clone = other.clone();
        let out_clone = out.clone();
        out.0.borrow_mut().backward = Box::new(move || {
            let out_grad = out_clone.0.borrow().grad.clone();

            // Every output element corresponds directly to one element of
            // Left matrix, so it's local gradient is uncahnged.
            let self_delta = out_grad.clone();

            let other_delta = match broadcast_columns {
                Some(columns) => {
                    let mut delta = vec![0.0; columns];

                    // The broadcast row is reused for every matrix row. Gradient
                    // controbutions therefore reduce into their corresponding column.
                    for (index, grad) in out_grad.iter().enumerate() {
                        delta [index % columns] += grad;
                    }

                    delta

                }
                None => out_grad,

            };

            // Compute all deltas before mutation and update each RefCell in a
            // seperate scope. This also preserves reused-node accumulation.
            {
                let mut self_inner = self_clone.0.borrow_mut();

                for (grad, delta) in self_inner.grad.iter_mut().zip(self_delta.iter()) {
                    *grad += *delta;
                }
            }

            {
                let mut other_inner = other_clone.0.borrow_mut();

                for (grad, delta) in other_inner.grad.iter_mut().zip(other_delta.iter()) {
                    *grad += *delta;
                }
            }
        });

        out
    }

    fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.0.borrow().shape,
            other.0.borrow().shape,
            "mul: shape mismatch"
        );

        let data: Vec<f64> = self
            .0
            .borrow()
            .data
            .iter()
            .zip(other.0.borrow().data.iter())
            .map(|(a, b)| a * b)
            .collect();

        let shape = self.0.borrow().shape.clone();
        let out = Tensor::from_op(data, shape, vec![self.clone(), other.clone()]);

        let self_clone = self.clone();
        let other_clone = other.clone();
        let out_clone = out.clone();
        out.0.borrow_mut().backward = Box::new(move || {
            let out_grad = out_clone.0.borrow().grad.clone();
            let self_data = self_clone.0.borrow().data.clone();
            let other_data = other_clone.0.borrow().data.clone();

            let mut self_grad_delta = vec![0.0; out_grad.len()];
            let mut other_grad_delta = vec![0.0; out_grad.len()];
            for i in 0..out_grad.len() {
                self_grad_delta[i] = out_grad[i] * other_data[i];
                other_grad_delta[i] = out_grad[i] * self_data[i];
            }
            for (i, delta) in self_grad_delta.iter().enumerate() {
                self_clone.0.borrow_mut().grad[i] += delta;
            }
            for (i, delta) in other_grad_delta.iter().enumerate() {
                other_clone.0.borrow_mut().grad[i] += delta;
            }
        });

        out
    }

    fn matmul(&self, other: &Tensor) -> Tensor {
        let left_shape = self.0.borrow().shape.clone();
        let right_shape = other.0.borrow().shape.clone();

        assert_eq!(
            self.0.borrow().shape.len(),
            2,
            "matmul: left tensor must be 2-D"
        );
        assert_eq!(
            other.0.borrow().shape.len(),
            2,
            "matmul: right tensor must be 2-D"
        );

        // Matrix multiplcation requires [m, k] x [k, n].
        let rows = left_shape[0];
        let inner = left_shape[1];
        let other_inner = right_shape[0];
        let cols = right_shape[1];

        assert_eq!(inner, other_inner, "matmul: inner dimensions must match");

        let left_data = self.0.borrow().data.clone();
        let right_data = other.0.borrow().data.clone();

        let mut data = vec![0.0; rows * cols];

        // Row-major matrix multiplicaiton:
        // output[row, col] is the dot product of a left row and right column.
        for row in 0..rows {
            for col in 0..cols {
                let mut sum = 0.0;

                for k in 0..inner {
                    let left_index = row * inner + k;
                    let right_index = k * cols + col;

                    sum += left_data[left_index] * right_data[right_index]
                }

                let output_index = row * cols + col;
                data[output_index] = sum;
            }
        }

        let out = Tensor::from_op(data, vec![rows, cols], vec![self.clone(), other.clone()]);

        let self_clone = self.clone();
        let out_clone = out.clone();
        let other_clone = other.clone();

        out.0.borrow_mut().backward = Box::new(move || {
            let out_grad = out_clone.0.borrow().grad.clone();
            let mut left_grad_delta = vec![0.0; rows * inner];
            let mut right_grad_delta = vec![0.0; inner * cols];

            // dA = dC x Bᵀ
            for row in 0..rows {
                for k in 0..inner {
                    let mut sum = 0.0;

                    for col in 0..cols {
                        let out_grad_index = row * cols + col;
                        let right_index = k * cols + col;

                        sum += out_grad[out_grad_index] * right_data[right_index];
                    }

                    let left_grad_index = row * inner + k;
                    left_grad_delta[left_grad_index] = sum;
                }
            }

            // dB = Aᵀ x dC
            for k in 0..inner {
                for col in 0..cols {
                    let mut sum = 0.0;

                    for row in 0..rows {
                        let left_index = row * inner + k;
                        let out_grad_index = row * cols + col;

                        sum += left_data[left_index] * out_grad[out_grad_index];
                    }

                    let right_grad_index = k * cols + col;
                    right_grad_delta[right_grad_index] = sum;
                }
            }

            // Apply the completed delta buffer after all reads are finished.
            for (i, delta) in left_grad_delta.iter().enumerate() {
                self_clone.0.borrow_mut().grad[i] += delta;
            }
            for (i, delta) in right_grad_delta.iter().enumerate() {
                other_clone.0.borrow_mut().grad[i] += delta;
            }
        });

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_with_matching_shape() {
        let t = TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t.data.len(), 6);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.grad, vec![0.0; 6]);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn rejects_mismatched_shape() {
        // Using 5 numbers but the shape should require 6. This should give us an error.
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);
    }

    #[test]
    #[should_panic(expected = "add: shape mismatch")]
    fn add_rejects_mismatched_shapes() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new(vec![3.0, 4.0], vec![1, 2]);

        a.add(&b);
    }

    #[test]
    #[should_panic(expected = "mul: shape mismatch")]
    fn mul_rejects_mismatched_shapes() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new(vec![3.0, 4.0], vec![1, 2]);

        a.mul(&b);
    }

    #[test]
    fn tensor_new_wraps_validated_data() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.0.borrow().data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.0.borrow().shape, vec![2, 2]);
    }

    #[test]
    fn add_forward_and_backward() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);

        let c = a.add(&b);

        // Forward elentwise sum
        assert_eq!(c.0.borrow().data, vec![11.0, 22.0, 33.0, 44.0]);
        assert_eq!(c.0.borrow().shape, vec![2, 2]);

        // Seed the output manually
        c.0.borrow_mut().grad = vec![1.0, 1.0, 1.0, 1.0];
        (c.0.borrow().backward)();

        // Passes gradient straight through. Grad = 1
        assert_eq!(a.0.borrow().grad, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(b.0.borrow().grad, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn mul_forward_and_backward() {
        let a = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0], vec![3]);

        let c = a.mul(&b);

        // Forward elemntwise product
        assert_eq!(c.0.borrow().data, vec![10.0, 18.0, 28.0]);
        assert_eq!(c.0.borrow().shape, vec![3]);

        // Seed grad to 1s
        c.0.borrow_mut().grad = vec![1.0, 1.0, 1.0];
        (c.0.borrow().backward)();

        // Product rule
        assert_eq!(a.0.borrow().grad, vec![5.0, 6.0, 7.0]);
        assert_eq!(b.0.borrow().grad, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn add_reused_tensor_accumulates() {
        let a = Tensor::new(vec![3.0, 4.0], vec![2]);

        let y = a.add(&a);

        y.0.borrow_mut().grad = vec![1.0, 1.0];
        (y.0.borrow().backward)();

        assert_eq!(a.0.borrow().grad, vec![2.0, 2.0]);
    }

    #[test]
    fn mul_reused_tensor_accumulates() {
        let a = Tensor::new(vec![3.0, 4.0], vec![2]);

        let y = a.mul(&a);

        y.0.borrow_mut().grad = vec![1.0, 1.0];
        (y.0.borrow().backward)();

        assert_eq!(a.0.borrow().grad, vec![6.0, 8.0]);
    }

    #[test]
    fn add_records_input_tensors_as_children() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new(vec![3.0, 4.0], vec![2]);

        let output = a.add(&b);

        let output_data = output.0.borrow();

        assert_eq!(output_data.children.len(), 2);
        assert!(std::rc::Rc::ptr_eq(&output_data.children[0].0, &a.0));
        assert!(std::rc::Rc::ptr_eq(&output_data.children[1].0, &b.0));
    }

    #[test]
    fn mul_records_input_tensors_as_children() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new(vec![3.0, 4.0], vec![2]);

        let output = a.mul(&b);
        let output_data = output.0.borrow();

        assert_eq!(output_data.children.len(), 2);
        assert!(Rc::ptr_eq(&output_data.children[0].0, &a.0));
        assert!(Rc::ptr_eq(&output_data.children[1].0, &b.0));
    }

    #[test]
    fn matmul_forward_2x3_by_3x2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

        let c = a.matmul(&b);

        assert_eq!(c.0.borrow().data, vec![58.0, 64.0, 139.0, 154.0]);
        assert_eq!(c.0.borrow().shape, vec![2, 2]);
    }

    #[test]
    #[should_panic(expected = "matmul: inner dimensions must match")]
    fn matmul_rejects_mismatched_inner_dimensions() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0], vec![2, 2]);

        a.matmul(&b);
    }

    #[test]
    fn matmul_records_input_tensors_as_children() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

        let output = a.matmul(&b);
        let output_data = output.0.borrow();

        assert_eq!(output_data.children.len(), 2);
        assert!(Rc::ptr_eq(&output_data.children[0].0, &a.0));
        assert!(Rc::ptr_eq(&output_data.children[1].0, &b.0));
    }

    #[test]
    #[should_panic(expected = "matmul: left tensor must be 2-D")]
    fn matmul_rejects_non_matrix_left_operand() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![3, 2]);

        a.matmul(&b);
    }

    #[test]
    #[should_panic(expected = "matmul: right tensor must be 2-D")]
    fn matmul_rejects_non_matrix_right_operand() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let b = Tensor::new(vec![7.0, 8.0, 9.0], vec![3]);

        a.matmul(&b);
    }

    #[test]
    fn matmul_backward_with_unit_output_gradient() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

        let c = a.matmul(&b);

        // Seed dC with ones, then run matmul's local backward rule.
        c.0.borrow_mut().grad = vec![1.0, 1.0, 1.0, 1.0];
        (c.0.borrow().backward)();

        assert_eq!(a.0.borrow().grad, vec![15.0, 19.0, 23.0, 15.0, 19.0, 23.0],);
        assert_eq!(b.0.borrow().grad, vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0],);
    }

    #[test]
    fn matmul_backward_with_non_uniform_output_gradient() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

        let c = a.matmul(&b);

        // dC = [[1, 2], [3, 4]]
        c.0.borrow_mut().grad = vec![1.0, 2.0, 3.0, 4.0];
        (c.0.borrow().backward)();

        assert_eq!(a.0.borrow().grad, vec![23.0, 29.0, 35.0, 53.0, 67.0, 81.0]);
        assert_eq!(b.0.borrow().grad, vec![13.0, 18.0, 17.0, 24.0, 21.0, 30.0]);
    }

    #[test]
    fn matmul_reused_tensor_accumulates_gradients() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let c = a.matmul(&a);

        c.0.borrow_mut().grad = vec![1.0, 1.0, 1.0, 1.0];
        (c.0.borrow().backward)();

        assert_eq!(a.0.borrow().grad, vec![7.0, 11.0, 9.0, 13.0]);
    }

    #[test]
    fn add_broadcast_right_row_vector_across_matrix_rows() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let row = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);

        let output = matrix.add(&row);
        let output_data = output.0.borrow();

        assert_eq!(output_data.data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        assert_eq!(output_data.shape, vec![2, 3]);
    }

    #[test]
    fn add_broadcast_right_row_vector_reduces_gradient_across_rows() {
        let matrix = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        let row = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);

        let output = matrix.add(&row);
         output.0.borrow_mut().grad = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
         (output.0.borrow().backward)();

         assert_eq!(row.0.borrow().grad, vec![5.0, 7.0, 9.0]);
    }
}
