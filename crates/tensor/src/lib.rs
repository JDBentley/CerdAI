// TODO:
// [x] TensorData - add flat data and shape with the size invariant
// Indexing - add in the newtype for TensorData handle
// Add - create elementwise
// Mul - create elementwise
// Backward - topological walk, seed, and run closures

struct TensorData {
    data: Vec<f64>,
    shape: Vec<usize>,
}

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
        TensorData { data, shape }
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
    fn rejects_mitsmatched_shape(){
        // Using 5 numbers but the shape should require 6. This should give us an error.
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);
    }
}