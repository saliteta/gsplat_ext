import glob
import json
import os
import shutil
from subprocess import DEVNULL, call

from rich.console import Console
from torch.utils.cpp_extension import (
    _get_build_directory,
    _import_module_from_library,
    load,
    CUDA_HOME as TORCH_CUDA_HOME,
)

PATH = os.path.dirname(os.path.abspath(__file__))
NO_FAST_MATH = os.getenv("NO_FAST_MATH", "0") == "1"
MAX_JOBS = os.getenv("MAX_JOBS")
need_to_unset_max_jobs = False
if not MAX_JOBS:
    need_to_unset_max_jobs = True
    os.environ["MAX_JOBS"] = "10"


def load_extension(
    name,
    sources,
    extra_cflags=None,
    extra_cuda_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    build_directory=None,
):
    """Load a JIT compiled extension."""
    if build_directory:
        os.makedirs(build_directory, exist_ok=True)
    try:
        return load(
            name,
            sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=extra_include_paths,
            build_directory=build_directory,
        )
    except OSError:
        return _import_module_from_library(name, build_directory, True)


def cuda_toolkit_available():
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False


def cuda_toolkit_version():
    cuda_home = os.path.join(os.path.dirname(shutil.which("nvcc")), "..")
    if os.path.exists(os.path.join(cuda_home, "version.txt")):
        with open(os.path.join(cuda_home, "version.txt")) as f:
            cuda_version = f.read().strip().split()[-1]
    elif os.path.exists(os.path.join(cuda_home, "version.json")):
        with open(os.path.join(cuda_home, "version.json")) as f:
            cuda_version = json.load(f)["cuda"]["version"]
    else:
        raise RuntimeError("Cannot find the cuda version.")
    return cuda_version


_C = None

try:
    # try to import the compiled module (via setup.py)
    from gsplat_ext import csrc as _C
except ImportError:
    # if failed, try with JIT compilation
    if cuda_toolkit_available():
        name = "gsplat_ext"
        build_dir = _get_build_directory(name, verbose=False)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        glm_path = os.path.join(current_dir, "csrc", "third_party", "glm")

        # --- NEW: find CUDA include/lib directories -------------------------
        cuda_home = os.environ.get("CUDA_HOME") or TORCH_CUDA_HOME
        # fall back to CONDA_PREFIX when installed via conda
        if not cuda_home:
            cuda_home = os.environ.get("CONDA_PREFIX")

        cuda_include = None
        candidate_includes = []
        if cuda_home:
            candidate_includes = [
                os.path.join(cuda_home, "include"),
                os.path.join(cuda_home, "targets", "x86_64-linux", "include"),  # conda layout
            ]
            for inc in candidate_includes:
                if os.path.exists(os.path.join(inc, "cuda_runtime_api.h")):
                    cuda_include = inc
                    break
        if not cuda_include:
            raise RuntimeError(
                "Could not find 'cuda_runtime_api.h'. "
                "Set CUDA_HOME or ensure CUDA Toolkit headers are installed. "
                f"Tried: {candidate_includes or 'N/A'}"
            )

        cuda_lib = os.path.join(cuda_home, "lib64")
        if not os.path.isdir(cuda_lib):
            alt = os.path.join(cuda_home, "lib")
            if os.path.isdir(alt):
                cuda_lib = alt
            else:
                cuda_lib = None
        # --------------------------------------------------------------------

        extra_include_paths = [
            os.path.join(PATH, "include/"),
            glm_path,
            cuda_include,  # <— key addition
        ]
        extra_ldflags = []
        if cuda_lib:
            extra_ldflags.append(f"-L{cuda_lib}")

        extra_cflags = ["-O3"]
        if NO_FAST_MATH:
            extra_cuda_cflags = ["-O3"]
        else:
            extra_cuda_cflags = ["-O3", "--use_fast_math"]
        extra_cuda_cflags += [
            "--expt-relaxed-constexpr",
            "--diag-suppress=20012",  # GLM defaulted ctor warnings
            "--diag-suppress=20011",
        ]

        sources = list(glob.glob(os.path.join(PATH, "csrc/*.cu"))) + list(
            glob.glob(os.path.join(PATH, "csrc/*.cpp"))
        )

        # Ensure at least one .cu is present so nvcc is invoked
        assert any(s.endswith(".cu") for s in sources), "No .cu sources found."

        try:
            os.remove(os.path.join(build_dir, "lock"))
        except OSError:
            pass

        if os.path.exists(os.path.join(build_dir, "gsplat_ext.so")) or os.path.exists(
            os.path.join(build_dir, "gsplat_ext.lib")
        ):
            _C = load_extension(
                name=name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_ldflags=extra_ldflags,
                extra_include_paths=extra_include_paths,
                build_directory=build_dir,
            )
        else:
            shutil.rmtree(build_dir, ignore_errors=True)
            with Console().status(
                f"[bold yellow]gsplat_ext: Setting up CUDA with MAX_JOBS={os.environ['MAX_JOBS']} (first build may take a bit)",
                spinner="bouncingBall",
            ):
                _C = load_extension(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_ldflags=extra_ldflags,
                    extra_include_paths=extra_include_paths,
                    build_directory=build_dir,
                )
    else:
        Console().print(
            "[yellow]gsplat_ext: No CUDA toolkit found. gsplat will be disabled.[/yellow]"
        )

if need_to_unset_max_jobs:
    os.environ.pop("MAX_JOBS")

__all__ = ["_C"]
