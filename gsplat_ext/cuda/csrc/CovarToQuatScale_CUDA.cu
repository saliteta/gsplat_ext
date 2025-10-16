#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h> 
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "Common.h"            // defines vec3, vec4, mat3 as float GLM types + CHECK_* macros
#include "CovarToQuatScale.h"  // (header for this TU)


#include <iostream>
#include <sstream>
#include <vector>
#include <c10/util/TypeCast.h>   // for c10::toString

static inline std::string shape_str(const at::Tensor& t) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < t.dim(); ++i) {
        oss << t.size(i);
        if (i + 1 < t.dim()) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

static inline void print_tensor_info(const char* name, const at::Tensor& t) {
    std::cout << name
              << " shape=" << shape_str(t)
              << " dtype=" << c10::toString(t.scalar_type())
              << " device=" << t.device().str()
              << " contiguous=" << (t.is_contiguous() ? "true" : "false")
              << std::endl;
    std::cout.flush();
}

namespace gsplat_ext {
namespace cg = cooperative_groups;

// -------------------- Device helpers (float-only) -------------------------

// cov6: [Σ00, Σ01, Σ02, Σ11, Σ12, Σ22]  (row-major upper)
static inline __device__ mat3 unpack_triu6_to_mat3(const float* __restrict__ cov6) {
    mat3 M(0.0f);
    // GLM is column-major: M[col][row]
    M[0][0] = cov6[0]; // Σ00
    M[1][1] = cov6[3]; // Σ11
    M[2][2] = cov6[5]; // Σ22
    M[1][0] = M[0][1] = cov6[1]; // Σ01
    M[2][0] = M[0][2] = cov6[2]; // Σ02
    M[2][1] = M[1][2] = cov6[4]; // Σ12
    return M;
}

// One Jacobi rotation on (p,q)
static inline __device__ void jacobiRotate(mat3& A, mat3& V, int p, int q) {
    float apq = A[p][q];
    if (fabsf(apq) < 1e-20f) return;

    float app = A[p][p], aqq = A[q][q];
    float tau = (aqq - app) / (2.0f * apq);
    float t   = copysignf(1.0f, tau) / (fabsf(tau) + sqrtf(1.0f + tau * tau));
    float c   = rsqrtf(1.0f + t * t);
    float s   = t * c;

    // A' = J^T A J
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        float aik = A[k][p], aiq = A[k][q];
        A[k][p] = c * aik - s * aiq;
        A[k][q] = s * aik + c * aiq;
    }
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        float akp = A[p][k], akq = A[q][k];
        A[p][k] = c * akp - s * akq;
        A[q][k] = s * akp + c * akq;
    }
    A[p][q] = A[q][p] = 0.0f;

    // Accumulate V <- V J
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        float vip = V[k][p], viq = V[k][q];
        V[k][p] = c * vip - s * viq;
        V[k][q] = s * vip + c * viq;
    }
}

// Symmetric eigendecomposition  A = V diag(e) V^T  (few sweeps)
static inline __device__ void eigenSym3(const mat3& Ain, vec3& evals, mat3& V) {
    mat3 A = Ain;     // already symmetric
    V = mat3(1.0f);   // identity
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        jacobiRotate(A, V, 0, 1);
        jacobiRotate(A, V, 0, 2);
        jacobiRotate(A, V, 1, 2);
    }
    evals = vec3(A[0][0], A[1][1], A[2][2]);
}

// Sort eigenvalues descending and permute columns of V accordingly
static inline __device__ void sortEigenDesc(vec3& e, mat3& V) {
    for (int i = 0; i < 3; ++i) {
        int m = i;
        for (int j = i + 1; j < 3; ++j)
            if ((&e.x)[j] > (&e.x)[m]) m = j;
        if (m != i) {
            float tmp = (&e.x)[i];
            (&e.x)[i] = (&e.x)[m];
            (&e.x)[m] = tmp;
            // swap columns i <-> m in V (column-major)
            for (int r = 0; r < 3; ++r) {
                float tr = V[r][i];
                V[r][i] = V[r][m];
                V[r][m] = tr;
            }
        }
    }
}

// Ensure right-handed rotation (det > 0)
static inline __device__ void makeProper(mat3& R) {
    if (glm::determinant(R) < 0.0f) {
        R[0][2] = -R[0][2];
        R[1][2] = -R[1][2];
        R[2][2] = -R[2][2];
    }
}

// Rotation matrix -> quaternion (w,x,y,z), normalize, enforce w>=0
static inline __device__ vec4 rotmat_to_quat(const mat3& R) {
    vec4 q; // (w,x,y,z)
    float tr = R[0][0] + R[1][1] + R[2][2];
    if (tr > 0.0f) {
        float t = sqrtf(tr + 1.0f) * 2.0f;
        q.x = (R[2][1] - R[1][2]) / t;
        q.y = (R[0][2] - R[2][0]) / t;
        q.z = (R[1][0] - R[0][1]) / t;
        q.w = 0.25f * t;
    } else {
        int i = 0;
        if (R[1][1] > R[0][0]) i = 1;
        if (R[2][2] > ((i==0) ? R[0][0] : R[1][1])) i = 2;
        if (i == 0) {
            float t = sqrtf(1.0f + R[0][0] - R[1][1] - R[2][2]) * 2.0f;
            q.w = (R[2][1] - R[1][2]) / t;
            q.x = 0.25f * t;
            q.y = (R[0][1] + R[1][0]) / t;
            q.z = (R[0][2] + R[2][0]) / t;
        } else if (i == 1) {
            float t = sqrtf(1.0f + R[1][1] - R[0][0] - R[2][2]) * 2.0f;
            q.w = (R[0][2] - R[2][0]) / t;
            q.x = (R[0][1] + R[1][0]) / t;
            q.y = 0.25f * t;
            q.z = (R[1][2] + R[2][1]) / t;
        } else {
            float t = sqrtf(1.0f + R[2][2] - R[0][0] - R[1][1]) * 2.0f;
            q.w = (R[1][0] - R[0][1]) / t;
            q.x = (R[0][2] + R[2][0]) / t;
            q.y = (R[1][2] + R[2][1]) / t;
            q.z = 0.25f * t;
        }
    }
    float invn = rsqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    q *= invn;
    if (q.w < 0.0f) q = -q;
    return q;
}

// Σ (packed) -> (quat, scale)
static inline __device__ void covar_to_quat_scale_device(
    const float* __restrict__ cov6,
    vec4& quat,  // (w,x,y,z)
    vec3& scale, // (sx,sy,sz) descending
    const float eps = 1e-8f
){
    mat3 Sigma = unpack_triu6_to_mat3(cov6);

    vec3 evals;
    mat3 V;
    eigenSym3(Sigma, evals, V);

    evals.x = fmaxf(evals.x, eps);
    evals.y = fmaxf(evals.y, eps);
    evals.z = fmaxf(evals.z, eps);

    sortEigenDesc(evals, V);

    scale = vec3(sqrtf(evals.x), sqrtf(evals.y), sqrtf(evals.z));

    makeProper(V);
    quat = rotmat_to_quat(V);
}

// === Non-cooperative, grid-strided kernel =================================
__global__ void covar_to_quat_scale_kernel_strided(
    uint32_t N,
    const float* __restrict__ covars6, // [N,6]
    float* __restrict__ quats,         // [N,4] (w,x,y,z)
    float* __restrict__ scales         // [N,3]
){
    uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t idx = tid; idx < N; idx += stride) {
        const float* cov6 = covars6 + idx * 6;
        vec4 q; vec3 s;
        covar_to_quat_scale_device(cov6, q, s);

        float* qout = quats  + idx * 4;
        float* sout = scales + idx * 3;
        qout[0]=q.w; qout[1]=q.x; qout[2]=q.y; qout[3]=q.z;
        sout[0]=s.x; sout[1]=s.y; sout[2]=s.z;
    }
}

// === Cooperative version (optional) =======================================
__global__ void covar_to_quat_scale_kernel_cg(
    uint32_t N,
    const float* __restrict__ covars6, // [N,6]
    float* __restrict__ quats,         // [N,4]
    float* __restrict__ scales         // [N,3]
){
    cg::grid_group grid = cg::this_grid();
    // one guarded print (ok to remove)
    if (grid.thread_rank() == 0) {
        printf("[covar_to_quat_scale] CG kernel start, N=%u\n", N);
    }
    for (uint32_t idx = grid.thread_rank(); idx < N; idx += grid.size()) {
        const float* cov6 = covars6 + idx * 6;
        vec4 q; vec3 s;
        covar_to_quat_scale_device(cov6, q, s);
        float* qout = quats  + idx * 4;
        float* sout = scales + idx * 3;
        qout[0]=q.w; qout[1]=q.x; qout[2]=q.y; qout[3]=q.z;
        sout[0]=s.x; sout[1]=s.y; sout[2]=s.z;
    }
}

// === Launcher ==============================================================
void launch_covar_to_quat_scale_kernel(
    const at::Tensor covars,  // [N,6] float32 CUDA contiguous
    at::Tensor quats,         // [N,4] float32 CUDA contiguous
    at::Tensor scales         // [N,3] float32 CUDA contiguous
){
    CHECK_INPUT(covars);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    TORCH_CHECK(covars.dtype()==at::kFloat && quats.dtype()==at::kFloat && scales.dtype()==at::kFloat,
                "covars/quats/scales must be float32");
    TORCH_CHECK(covars.dim()==2 && covars.size(1)==6, "covars must be [N,6]");

    const uint32_t N = static_cast<uint32_t>(covars.size(0));
    if (N == 0) return;

    auto torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    const int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);
    blocks = std::max(1, std::min(blocks, 65535));

    // ---- default path: normal strided launch (rock-solid) ----
    auto do_normal = [&](){
        covar_to_quat_scale_kernel_strided<<<blocks, threads, 0, stream>>>(
            N,
            covars.data_ptr<float>(),
            quats.data_ptr<float>(),
            scales.data_ptr<float>());
        AT_CUDA_CHECK(cudaGetLastError());
    };

    // toggle cooperative via env
    const char* use_cg_env = std::getenv("GSPLAT_USE_CG");
    const bool request_cg = (use_cg_env && *use_cg_env=='1');

    if (!request_cg) {
        do_normal();
        return;
    }

    // ---- optional: try cooperative launch, fall back on any error ----
    int dev = 0;
    AT_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    AT_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    if (!prop.cooperativeLaunch) {
        fprintf(stderr, "[covar_to_quat_scale] cooperativeLaunch not supported on this device; using normal launch.\n");
        do_normal();
        return;
    }

    // cap blocks to cooperative capacity
    int blocksPerSM = 0;
    AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM, covar_to_quat_scale_kernel_cg, threads, 0));
    int maxCoopBlocks = std::max(1, blocksPerSM * prop.multiProcessorCount);
    int coopBlocks = std::max(1, std::min(blocks, maxCoopBlocks));

    void* args[] = {
        (void*)&N,
        (void*)covars.data_ptr<float>(),
        (void*)quats.data_ptr<float>(),
        (void*)scales.data_ptr<float>()
    };

    cudaError_t st = cudaLaunchCooperativeKernel(
        (void*)covar_to_quat_scale_kernel_cg,
        dim3(coopBlocks), dim3(threads), args, 0, stream);

    if (st != cudaSuccess) {
        fprintf(stderr,
            "[covar_to_quat_scale] cooperative launch failed: %s (req blocks=%d, cap=%d, threads=%d, SMs=%d, blocks/SM=%d)\n",
            cudaGetErrorString(st), blocks, maxCoopBlocks, threads, prop.multiProcessorCount, blocksPerSM);
        // fall back safely
        do_normal();
        return;
    }

    AT_CUDA_CHECK(cudaGetLastError());

}
}