#![feature(test)]

extern crate test;
extern crate nalgebra;

use std::ops::{Mul, Index};

macro_rules! unroll_sum_4 (
    ($f: expr) => (
        ($f)(0) + ($f)(1) + ($f)(2) + ($f)(3)
    )
);

macro_rules! unroll_Mat4x4(
    ($f: expr) => (
        Mat4x4(
            [[($f)(0, 0), ($f)(0, 1), ($f)(0, 2), ($f)(0, 3)],
             [($f)(1, 0), ($f)(1, 1), ($f)(1, 2), ($f)(1, 3)],
             [($f)(2, 0), ($f)(2, 1), ($f)(2, 2), ($f)(2, 3)],
             [($f)(3, 0), ($f)(3, 1), ($f)(3, 2), ($f)(3, 3)]]
        )
    )
);

#[derive(Debug, Copy, Clone)]
pub struct Mat4x4(pub [[f64; 4]; 4]);

impl Index<usize> for Mat4x4 {
    type Output = [f64; 4];
    #[inline(always)]
    fn index<'a>(&'a self, i: usize) -> &'a [f64; 4] {
        self.0.index(i)
    }
}

impl Mul<Mat4x4> for Mat4x4 {
    type Output = Mat4x4;
    #[inline(always)]
    fn mul(self, rhs: Mat4x4) -> Mat4x4 {
        unroll_Mat4x4!(|i, j| unroll_sum_4!(|k| self[i][k] * rhs[k][j]))
    }
}

impl Mat4x4 {
    #[inline(always)]
    fn t(&self) -> Self {
        unroll_Mat4x4!(|i, j| self[j][i])
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use test;
    use nalgebra::*;

    fn nalgebra_4x4s() -> (Matrix4<f64>, Matrix4<f64>) {
        (Matrix4::new(
            1., 1., 1., 1.,
            1., 2., 1., 1.,
            1., 1., 4., 1.,
            1., 1., 1., 1.
        ), Matrix4::new(
            1., 0., 0., 0.,
            0., 1., 0., 0.,
            0., 0., 1., 0.,
            0., 0., 0., 1.,
        ))
    }

    fn unrolled_4x4s() -> (Mat4x4, Mat4x4) {
        (Mat4x4(
            [[1., 1., 1., 1.]
            ,[1., 2., 1., 1.]
            ,[1., 1., 4., 1.]
            ,[1., 1., 1., 1.]]
        ), Mat4x4(
            [[1., 0., 0., 0.]
            ,[0., 1., 0., 0.]
            ,[0., 0., 1., 0.]
            ,[0., 0., 0., 1.]]
        ))
    }

    #[bench]
    fn bench_4x4_mult_nalgebra(b: &mut test::Bencher) {
        b.iter(|| {
            let (mut a, b) = test::black_box(nalgebra_4x4s());
            for _ in 0..1_000 {
                a = a * b;
            }
            test::black_box(a);
        });
    }

    #[bench]
    fn bench_4x4_mult_unrolled(b: &mut test::Bencher) {
        b.iter(|| {
            let (mut a, b) = test::black_box(unrolled_4x4s());
            for _ in 0..1_000 {
                a = a * b;
            }
            test::black_box(a);
        });
    }

    #[bench]
    fn bench_4x4_t_mult_nalgebra(b: &mut test::Bencher) {
        b.iter(|| {
            let (mut a, b) = test::black_box(nalgebra_4x4s());
            for _ in 0..1_000 {
                a = a.transpose() * b;
            }
            test::black_box(a);
        });
    }

    #[bench]
    fn bench_4x4_t_mult_unrolled(b: &mut test::Bencher) {
        b.iter(|| {
            let (mut a, b) = test::black_box(unrolled_4x4s());
            for _ in 0..1_000 {
                a = a.t() * b;
            }
            test::black_box(a);
        });
    }
}
