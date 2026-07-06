use std::rc::Rc;
use std::cell::RefCell;

struct ValueData {
    data: f64,
    grad: f64,
    children: Vec<Value>,
}

#[derive(Clone)]
struct Value(Rc<RefCell<ValueData>>);

impl Value {
    fn new(data: f64) -> Value {
        Value (Rc::new(RefCell::new(ValueData {
            data,
            grad: 0.0,
            children: Vec::new(),
        })))
    }

    fn add(&self, other: &Value) -> Value {
        let data = self.0.borrow().data + other.0.borrow().data;
        Value(Rc::new(RefCell::new(ValueData {
            data,
            grad: 0.0,
            children: vec![self.clone(), other.clone()],
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_a_value() {
        let v = Value::new(3.0);
        assert_eq!(v.0.borrow().data, 3.0);
        assert_eq!(v.0.borrow().grad, 0.0);
    }
    
    #[test]
    fn value_can_feed_two_ops() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a.add(&b);
        let d = a.add(&b);
        assert_eq!(c.0.borrow().data, 5.0);
        assert_eq!(d.0.borrow().data, 5.0);
    }
}
