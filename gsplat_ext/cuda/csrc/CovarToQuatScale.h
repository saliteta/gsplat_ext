#pragma once

#include <cstdint>

namespace at {
class Tensor;
}

namespace gsplat_ext {

void launch_covar_to_quat_scale_kernel(
    // inputs
    const at::Tensor covars,  // [N, 6]
    // outputs
    at::Tensor quats, // [N, 4]
    at::Tensor scales // [N, 3]
);

} // namespace gsplat_ext