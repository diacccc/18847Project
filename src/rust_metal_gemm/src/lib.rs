// src/rust_metal_gemm/src/lib.rs
use std::ffi::{c_float, c_void};
use std::slice;
use metal::{CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};

// Fixed Metal kernel with improved precision handling
const METAL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemm_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    constant float& beta [[buffer(4)]],
    constant int& M [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& K [[buffer(7)]],
    constant int& lda [[buffer(8)]],
    constant int& ldb [[buffer(9)]],
    constant int& ldc [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int i = gid.x;
    const int j = gid.y;
    
    // Bounds check
    if (i >= M || j >= N) return;
    
    // Use higher precision for accumulation
    float acc = 0.0f;
    
    // Calculate dot product
    for (int k = 0; k < K; k++) {
        // Column-major indexing: i + k * lda and k + j * ldb
        acc += A[i + k * lda] * B[k + j * ldb];
    }
    
    // Apply alpha scaling
    acc *= alpha;
    
    // Apply beta scaling and store result
    if (beta != 0.0f) {
        C[i + j * ldc] = acc + beta * C[i + j * ldc];
    } else {
        C[i + j * ldc] = acc;
    }
}
"#;

pub struct MetalGEMM {
    device: Device,
    command_queue: CommandQueue,
    pipeline_state: ComputePipelineState,
}

impl MetalGEMM {
    pub fn new() -> Self {
        let device = Device::system_default().expect("No Metal device found");
        let command_queue = device.new_command_queue();
        
        // Compile the Metal kernel
        let lib = device
            .new_library_with_source(METAL_SOURCE, &metal::CompileOptions::new())
            .expect("Failed to compile Metal source");
        let kernel = lib
            .get_function("gemm_kernel", None)
            .expect("Failed to get kernel function");
        let pipeline_state = device
            .new_compute_pipeline_state_with_function(&kernel)
            .expect("Failed to create pipeline state");
            
        Self {
            device,
            command_queue,
            pipeline_state,
        }
    }
    
    pub fn execute(
        &self,
        alpha: f32,
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        a_ld: usize,
        b: &[f32],
        b_rows: usize,
        b_cols: usize,
        b_ld: usize,
        beta: f32,
        c: &mut [f32],
        c_rows: usize,
        c_cols: usize,
        c_ld: usize,
    ) {
        // Verify dimensions match
        assert_eq!(a_cols, b_rows, "Matrix dimensions don't match: A.cols != B.rows");
        assert_eq!(a_rows, c_rows, "Matrix dimensions don't match: A.rows != C.rows");
        assert_eq!(b_cols, c_cols, "Matrix dimensions don't match: B.cols != C.cols");
        
        let m = a_rows;
        let n = b_cols;
        let k = a_cols;
        
        // Create Metal buffers with appropriate sizes
        let a_buffer = self.device.new_buffer_with_data(
            a.as_ptr() as *const c_void,
            (a_ld * a_cols * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let b_buffer = self.device.new_buffer_with_data(
            b.as_ptr() as *const c_void,
            (b_ld * b_cols * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let c_buffer = self.device.new_buffer_with_data(
            c.as_ptr() as *const c_void,
            (c_ld * c_cols * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Create scalar value buffers
        let alpha_buffer = self.device.new_buffer_with_data(
            &alpha as *const f32 as *const c_void,
            std::mem::size_of::<f32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let beta_buffer = self.device.new_buffer_with_data(
            &beta as *const f32 as *const c_void,
            std::mem::size_of::<f32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Create dimension buffers
        let m_buffer = self.device.new_buffer_with_data(
            &(m as i32) as *const i32 as *const c_void,
            std::mem::size_of::<i32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let n_buffer = self.device.new_buffer_with_data(
            &(n as i32) as *const i32 as *const c_void,
            std::mem::size_of::<i32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let k_buffer = self.device.new_buffer_with_data(
            &(k as i32) as *const i32 as *const c_void,
            std::mem::size_of::<i32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let lda_buffer = self.device.new_buffer_with_data(
            &(a_ld as i32) as *const i32 as *const c_void,
            std::mem::size_of::<i32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let ldb_buffer = self.device.new_buffer_with_data(
            &(b_ld as i32) as *const i32 as *const c_void,
            std::mem::size_of::<i32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let ldc_buffer = self.device.new_buffer_with_data(
            &(c_ld as i32) as *const i32 as *const c_void,
            std::mem::size_of::<i32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Create command buffer and compute encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        
        // Set compute pipeline state and buffers
        compute_encoder.set_compute_pipeline_state(&self.pipeline_state);
        compute_encoder.set_buffer(0, Some(&a_buffer), 0);
        compute_encoder.set_buffer(1, Some(&b_buffer), 0);
        compute_encoder.set_buffer(2, Some(&c_buffer), 0);
        compute_encoder.set_buffer(3, Some(&alpha_buffer), 0);
        compute_encoder.set_buffer(4, Some(&beta_buffer), 0);
        compute_encoder.set_buffer(5, Some(&m_buffer), 0);
        compute_encoder.set_buffer(6, Some(&n_buffer), 0);
        compute_encoder.set_buffer(7, Some(&k_buffer), 0);
        compute_encoder.set_buffer(8, Some(&lda_buffer), 0);
        compute_encoder.set_buffer(9, Some(&ldb_buffer), 0);
        compute_encoder.set_buffer(10, Some(&ldc_buffer), 0);
        
        // Configure grid and threadgroup sizes for optimal performance
        let threadgroup_size = MTLSize::new(16, 16, 1); // 16x16 threadgroup
        let grid_size = MTLSize::new(m.try_into().unwrap(), n.try_into().unwrap(), 1); // One thread per element
        
        // Dispatch threads
        compute_encoder.dispatch_threads(grid_size, threadgroup_size);
        compute_encoder.end_encoding();
        
        // Commit and wait for completion
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Copy results back to the host
        let ptr = c_buffer.contents() as *const f32;
        unsafe {
            // Make sure we only copy what we need
            for j in 0..c_cols {
                for i in 0..c_rows {
                    c[i + j * c_ld] = *ptr.add(i + j * c_ld);
                }
            }
        }
    }
}

// C interface for FFI
#[no_mangle]
pub extern "C" fn metal_gemm_create() -> *mut MetalGEMM {
    let gemm = Box::new(MetalGEMM::new());
    Box::into_raw(gemm)
}

#[no_mangle]
pub extern "C" fn metal_gemm_execute(
    gemm: *mut MetalGEMM,
    alpha: c_float,
    a: *const c_float,
    a_rows: usize,
    a_cols: usize,
    a_ld: usize,
    b: *const c_float,
    b_rows: usize,
    b_cols: usize,
    b_ld: usize,
    beta: c_float,
    c: *mut c_float,
    c_rows: usize,
    c_cols: usize,
    c_ld: usize,
) {
    let gemm = unsafe { &*gemm };
    
    let a_slice = unsafe { slice::from_raw_parts(a, a_rows * a_ld) };
    let b_slice = unsafe { slice::from_raw_parts(b, b_rows * b_ld) };
    let c_slice = unsafe { slice::from_raw_parts_mut(c, c_rows * c_ld) };
    
    gemm.execute(
        alpha, a_slice, a_rows, a_cols, a_ld,
        b_slice, b_rows, b_cols, b_ld,
        beta, c_slice, c_rows, c_cols, c_ld,
    );
}

#[no_mangle]
pub extern "C" fn metal_gemm_destroy(gemm: *mut MetalGEMM) {
    if !gemm.is_null() {
        unsafe {
            let _ = Box::from_raw(gemm);
        }
    }
}