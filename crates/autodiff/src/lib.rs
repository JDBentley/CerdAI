use std::rc::Rc;
use std::cell::RefCell;

pub struct ValueData {
    pub data: f64,
    pub grad: f64,
    children: Vec<Value>,
    backward: Box<dyn Fn()>,
}

#[derive(Clone)]
pub struct Value(pub Rc<RefCell<ValueData>>);

impl Value {
    pub fn new(data: f64) -> Value {
        Value (Rc::new(RefCell::new(ValueData {
            data,
            grad: 0.0,
            children: Vec::new(),
            backward: Box::new(|| {}),
        })))
    }

    pub fn add(&self, other: &Value) -> Value {
        let data = self.0.borrow().data + other.0.borrow().data;
        let out = Value(Rc::new(RefCell::new(ValueData {
            data,
            grad: 0.0,
            children: vec![self.clone(), other.clone()],
            backward: Box::new(|| {}),
        })));
        
        let self_clone = self.clone();
        let other_clone = other.clone();
        let out_clone = out.clone();
        out.0.borrow_mut().backward = Box::new(move || {
            self_clone.0.borrow_mut().grad += out_clone.0.borrow().grad;
            other_clone.0.borrow_mut().grad += out_clone.0.borrow().grad;
        });

        out
    }

    pub fn same_node(&self, other: &Value) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    pub fn build_topo(&self, visited: &mut Vec<Value>, topo: &mut Vec<Value>) {
        if visited.iter().any(|v| v.same_node(self)) {
            return;
        }
        visited.push(self.clone());
        for child in self.0.borrow().children.iter() {
            child.build_topo(visited, topo);
        }
        topo.push(self.clone());
    }

    pub fn backward(&self) {
        let mut visited = Vec::new();
        let mut topo = Vec::new();
        self.build_topo(&mut visited, &mut topo);

        self.0.borrow_mut().grad = 1.0;

        for node in topo.iter().rev() {
            (node.0.borrow().backward)();
        }
    }

    pub fn mul(&self, other: &Value) -> Value {
        let data = self.0.borrow().data * other.0.borrow().data;
        let out = Value(Rc::new(RefCell::new(ValueData {
            data,
            grad: 0.0,
            children: vec![self.clone(), other.clone()],
            backward: Box::new(|| {}),
        })));

        let self_clone = self.clone();
        let other_clone = other.clone();
        let out_clone = out.clone();
        out.0.borrow_mut().backward = Box::new(move || {
            let self_data = self_clone.0.borrow().data;
            let other_data = other_clone.0.borrow().data;
            let out_grad = out_clone.0.borrow().grad;
            self_clone.0.borrow_mut().grad += out_grad * other_data;
            other_clone.0.borrow_mut().grad += out_grad * self_data;
        });

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expr(a: f64, b: f64, c: f64) -> f64 {
        let av = Value::new(a);
        let bv = Value::new(b);
        let cv = Value::new(c);
        let out = av.mul(&bv).add(&cv);
        let result = out.0.borrow().data;
        result
    }

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

    #[test]
    fn single_add_backward() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a.add(&b);

        c.0.borrow_mut().grad = 1.0;
        (c.0.borrow().backward)();

        assert_eq!(a.0.borrow().grad, 1.0);
        assert_eq!(b.0.borrow().grad, 1.0);
    }

    #[test]
    fn same_node_is_identity_not_value() {
        let a = Value::new(2.0);
        let a_clone = a.clone();
        let b = Value::new(2.0);

        assert!(a.same_node(&a_clone));
        assert!(!a.same_node(&b));
    }

    #[test]
    fn composed_backward() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = Value::new(4.0);

        let d = a.add(&b);
        let e = d.add(&c);

        e.backward();

        assert_eq!(e.0.borrow().data, 9.0);
        assert_eq!(a.0.borrow().grad, 1.0);
        assert_eq!(b.0.borrow().grad, 1.0);
        assert_eq!(c.0.borrow().grad, 1.0);
        assert_eq!(d.0.borrow().grad, 1.0);
    }

    #[test]
    fn single_mul_backward() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a.mul(&b);

        c.backward();

        assert_eq!(c.0.borrow().data, 6.0);
        assert_eq!(a.0.borrow().grad, 3.0);
        assert_eq!(b.0.borrow().grad, 2.0);
    }
    
    #[test]
    fn gradient_accumulates_when_value_resused() {
        let a = Value::new(3.0);
        let y = a.mul(&a);

        y.backward();

        assert_eq!(y.0.borrow().data, 9.0);
        assert_eq!(a.0.borrow().grad, 6.0);
    }

    #[test]
    fn gradient_check_matches_finite_differences() {
        let (a, b, c) = (2.0, -3.0, 5.0);

        let av = Value::new(a);
        let bv = Value::new(b);
        let cv = Value::new(c);
        let out = av.mul(&bv).add(&cv);
        out.backward();

        let grad_a = av.0.borrow().grad;
        let grad_b = bv.0.borrow().grad;
        let grad_c = cv.0.borrow().grad;

        let h = 1e-5;
        let num_a = (expr(a + h, b, c) - expr(a - h, b, c)) / (2.0 * h);
        let num_b = (expr(a, b + h, c) - expr(a, b - h, c)) / (2.0 * h);
        let num_c = (expr(a, b, c + h) - expr(a, b, c - h)) / (2.0 * h);

        assert!((grad_a - num_a).abs() < 1e-6);
        assert!((grad_b - num_b).abs() < 1e-6);
        assert!((grad_c - num_c).abs() < 1e-6);
    }
}
