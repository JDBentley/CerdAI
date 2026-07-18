//! CPU tensor storage and automatic-differentation operations.
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
        assert_eq!(
            self.0.borrow().shape,
            other.0.borrow().shape,
            "add: shape mismatch"
        );

        let data: Vec<f64> = self
            .0
            .borrow()
            .data
            .iter()
            .zip(other.0.borrow().data.iter())
            .map(|(a, b)| a + b)
            .collect();

        let shape = self.0.borrow().shape.clone();
        let out = Tensor::from_op(data, shape, vec![self.clone(), other.clone()]);

        let self_clone = self.clone();
        let other_clone = other.clone();
        let out_clone = out.clone();
        out.0.borrow_mut().backward = Box::new(move || {
            let out_grad = out_clone.0.borrow().grad.clone();
            for (i, g) in out_grad.iter().enumerate() {
                self_clone.0.borrow_mut().grad[i] += g;
                other_clone.0.borrow_mut().grad[i] += g;
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
            for i in 0..out_grad.len() {
                self_clone.0.borrow_mut().grad[i] += self_grad_delta[i];
            }
            for i in 0..out_grad.len() {
                other_clone.0.borrow_mut().grad[i] += other_grad_delta[i];
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
    }

    #[test]
    #[should_panic]
    fn rejects_mismatched_shape() {
        // Using 5 numbers but the shape should require 6. This should give us an error.
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);
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

        // Seed grad to 1s
        c.0.borrow_mut().grad = vec![1.0, 1.0, 1.0];
        (c.0.borrow().backward)();

        // Product rule
        assert_eq!(a.0.borrow().grad, vec![5.0, 6.0, 7.0]);
        assert_eq!(b.0.borrow().grad, vec![2.0, 3.0, 4.0]);
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
}
