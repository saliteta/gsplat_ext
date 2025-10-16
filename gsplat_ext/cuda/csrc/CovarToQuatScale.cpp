#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include "Common.h"
#include "CovarToQuatScale.h"   // launcher decl
#include "Ops.h"                // this header declares covar_to_quat_scale(...)


namespace gsplat_ext {

std::tuple<at::Tensor, at::Tensor>
covar_to_quat_scale(const at::Tensor& covars) {   // *** const ref ***
    DEVICE_GUARD(covars);
    CHECK_INPUT(covars);
    const int64_t N = covars.size(0);
    auto opts = covars.options();
    at::Tensor quats  = at::empty({N, 4}, opts);
    at::Tensor scales = at::empty({N, 3}, opts);
    launch_covar_to_quat_scale_kernel(covars, quats, scales);
    return {quats, scales};
}

} // namespace gsplat_ext
