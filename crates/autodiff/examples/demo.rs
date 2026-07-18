use cerdai_autodiff::Value;

fn main() {
    let w = Value::new(2.0);
    let x = Value::new(3.0);
    let b = Value::new(-1.0);

    let output = w.mul(&x).add(&b);

    output.backward();

    println!("Neuron output = w * x + b");
    println!(
        "w = {}, x = {}, b = {}",
        w.0.borrow().data,
        x.0.borrow().data,
        b.0.borrow().data
    );
    println!("output = {}", output.0.borrow().data);
    println!();
    println!("Gradients (d output / d input):");
    println!("dw = {} (equals x)", w.0.borrow().grad);
    println!("dx = {} (equals w)", x.0.borrow().grad);
    println!("db = {} (equals 1)", b.0.borrow().grad);
}
