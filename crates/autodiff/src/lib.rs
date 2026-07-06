struct Value {
    data: f64,
    grad: f64,
    children: Vec<Value>,
}

impl Value {
    fn new(data: f64) -> Value {
        Value { data, grad: 0.0, children: Vec::new() }
    }

    fn add(self, other: Value) -> Value {
        Value {
            data: self.data + other.data,
            grad: 0.0,
            children: vec![self, other],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_a_value() {
        let v = Value::new(3.0);
        assert_eq!(v.data, 3.0);
        assert_eq!(v.grad, 0.0);
    }
    
    #[test]
    fn value_can_feed_two_ops() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a.add(b);
        let d = a.add(b);
        assert_eq!(c.data, 5.0);
        assert_eq!(d.data, 5.0);
    }
}