#![no_std]
use core::arch::wasm32::*;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

// Quantize one row of FP32 activations to int16 in place and return the scale so
// that x[i] ~= q[i] * scale. Ascending order is safe: q[i] (2 bytes at 2*i) only
// overwrites x[j] (4 bytes at 4*j) for j = i/2 < i, which was already read.
#[inline(always)]
unsafe fn quantize(x: *mut f32, cols: usize) -> f32 {
    let mut maxabs = 0.0f32;
    for i in 0..cols {
        let v = *x.add(i);
        let a = if v < 0.0 { -v } else { v };
        if a > maxabs { maxabs = a; }
    }
    let scale = if maxabs > 0.0 { maxabs / 32767.0 } else { 1.0 };
    let inverse = 1.0 / scale;
    let q = x as *mut i16;
    for i in 0..cols {
        let v = *x.add(i) * inverse;
        let rounded = if v >= 0.0 { v + 0.5 } else { v - 0.5 };
        *q.add(i) = rounded as i16;
    }
    scale
}

#[inline(always)]
fn horizontal(v: v128) -> f32 {
    f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v)
        + f32x4_extract_lane::<2>(v) + f32x4_extract_lane::<3>(v)
}

// int4 weights (group size 32, FP32 scale per group), int16 activations.
#[no_mangle]
pub unsafe extern "C" fn matmul_q4(
    weights: *const u8, scales: *const f32, inputs: *const f32,
    outputs: *mut f32, rows: usize, cols: usize, batch: usize,
) {
    for b in 0..batch {
        let x = inputs.add(b * cols) as *mut f32;
        let scale = quantize(x, cols);
        let xq = x as *const i16;
        for r in 0..rows {
            let mut acc = f32x4_splat(0.0);
            for g in 0..cols / 32 {
                let packed = v128_load(weights.add(r * cols / 2 + g * 16) as *const v128);
                let low = i8x16_sub(v128_and(packed, u8x16_splat(15)), i8x16_splat(8));
                let high = i8x16_sub(u8x16_shr(packed, 4), i8x16_splat(8));
                let a = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(low, high);
                let c = i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(low, high);
                let w0 = i16x8_extend_low_i8x16(a);
                let w1 = i16x8_extend_high_i8x16(a);
                let w2 = i16x8_extend_low_i8x16(c);
                let w3 = i16x8_extend_high_i8x16(c);
                let xp = xq.add(g * 32);
                let d0 = i32x4_dot_i16x8(w0, v128_load(xp as *const v128));
                let d1 = i32x4_dot_i16x8(w1, v128_load(xp.add(8) as *const v128));
                let d2 = i32x4_dot_i16x8(w2, v128_load(xp.add(16) as *const v128));
                let d3 = i32x4_dot_i16x8(w3, v128_load(xp.add(24) as *const v128));
                let sum = i32x4_add(i32x4_add(d0, d1), i32x4_add(d2, d3));
                let f = f32x4_convert_i32x4(sum);
                acc = f32x4_add(acc, f32x4_mul(f, f32x4_splat(*scales.add(r * (cols / 32) + g) * scale)));
            }
            *outputs.add(b * rows + r) = horizontal(acc);
        }
    }
}

// int8 weights (FP32 scale per row), int16 activations.
#[no_mangle]
pub unsafe extern "C" fn matmul(
    weights: *const i8, scales: *const f32, inputs: *const f32,
    outputs: *mut f32, rows: usize, cols: usize, batch: usize,
) {
    for b in 0..batch {
        let x = inputs.add(b * cols) as *mut f32;
        let scale = quantize(x, cols);
        let xq = x as *const i16;
        for r in 0..rows {
            let w = weights.add(r * cols);
            let mut acc = f32x4_splat(0.0);
            let mut c = 0;
            while c < cols {
                let bytes = v128_load(w.add(c) as *const v128);
                let w0 = i16x8_extend_low_i8x16(bytes);
                let w1 = i16x8_extend_high_i8x16(bytes);
                let xp = xq.add(c);
                let d0 = i32x4_dot_i16x8(w0, v128_load(xp as *const v128));
                let d1 = i32x4_dot_i16x8(w1, v128_load(xp.add(8) as *const v128));
                let f = f32x4_convert_i32x4(i32x4_add(d0, d1));
                acc = f32x4_add(acc, f32x4_mul(f, f32x4_splat(*scales.add(r))));
                c += 16;
            }
            *outputs.add(b * rows + r) = horizontal(acc) * scale;
        }
    }
}