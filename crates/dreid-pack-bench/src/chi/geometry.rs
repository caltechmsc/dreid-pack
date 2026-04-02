use std::f64::consts::PI;

pub fn wrap(angle: f64) -> f64 {
    let mut a = angle % (2.0 * PI);
    if a >= PI {
        a -= 2.0 * PI;
    } else if a < -PI {
        a += 2.0 * PI;
    }
    a
}

pub fn diff(a: f64, b: f64) -> f64 {
    let d = wrap(a - b).abs();
    if d > PI { 2.0 * PI - d } else { d }
}

pub fn dihedral(p0: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3], p3: &[f64; 3]) -> f64 {
    let b1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let b2 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let b3 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]];

    let n1 = cross(&b1, &b2);
    let n2 = cross(&b2, &b3);

    let m = cross(&n1, &b2_hat(&b2));

    let x = dot(&n1, &n2);
    let y = dot(&m, &n2);

    (-y).atan2(x)
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn b2_hat(v: &[f64; 3]) -> [f64; 3] {
    let len = dot(v, v).sqrt();
    if len == 0.0 {
        return [0.0; 3];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}
