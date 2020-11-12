#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
# Copyright (C) Broadcom. 2020-2020.  ALL RIGHTS RESERVED.
#

import os
import sys
from setuptools import setup
from torch.utils import cpp_extension

ucc_plugin_dir = os.path.dirname(os.path.abspath(__file__))
ucx_home = os.environ.get("UCX_HOME")
if ucx_home is None:
    ucx_home = os.environ.get("HPCX_UCX_DIR")
if ucx_home is None:
    print("Couldn't find UCX install dir, please set UCX_HOME env variable")
    sys.exit(1)

plugin_sources      = ["src/torch_ucc.cpp",
                       "src/torch_ucc_sendrecv.cpp",
                       "src/torch_ucx_alltoall.cpp",
                       "src/torch_ucx_coll.cpp"]
plugin_include_dirs = ["{}/include/".format(ucc_plugin_dir),
                       "{}/include/".format(ucx_home)]
plugin_library_dirs = ["{}/lib/".format(ucx_home)]
plugin_libraries    = ["ucp", "uct", "ucm", "ucs"]
plugin_compile_args = ['-g', '-O0']

with_xccl = os.environ.get("WITH_XCCL")
if with_xccl is None or with_xccl == "no":
    print("XCCL support is disabled")
else:
    print("XCCL support is enabled: {}".format(with_xccl))
    plugin_sources.append("src/torch_xccl.cpp")
    plugin_include_dirs.append("{}/include/".format(with_xccl))
    plugin_library_dirs.append("{}/lib/".format(with_xccl))
    plugin_libraries.append("xccl")
    plugin_compile_args.append("-DWITH_XCCL")

with_bnxt_co = os.environ.get("WITH_BNXT_CO")
if with_bnxt_co is None or with_bnxt_co == "no":
    print("bnxt_co offload support is disabled")
else:
    print("bnxt_co offload support is enabled: {}".format(with_bnxt_co))
    plugin_sources.append("src/torch_bnxt_co.cpp")
    plugin_include_dirs.append("{}/include/".format(with_bnxt_co))
    plugin_library_dirs.append("{}/lib/".format(with_bnxt_co))
    plugin_libraries.append("bnxtco")
    plugin_compile_args.append("-DWITH_BNXT_CO")

print(plugin_sources)
with_cuda = os.environ.get("WITH_CUDA")
if with_cuda is None or with_cuda == "no":
    print("CUDA support is disabled")
    module = cpp_extension.CppExtension(
        name = "torch_ucc",
        sources = plugin_sources,
        include_dirs = plugin_include_dirs,
        library_dirs = plugin_library_dirs,
        libraries = plugin_libraries,
        extra_compile_args=plugin_compile_args
    )
else:
    print("CUDA support is enabled")
    plugin_compile_args.append("-DUSE_CUDA")
    module = cpp_extension.CUDAExtension(
        name = "torch_ucc",
        sources = plugin_sources,
        include_dirs = plugin_include_dirs,
        library_dirs = plugin_library_dirs,
        libraries = plugin_libraries,
        extra_compile_args=plugin_compile_args
    )

setup(
    name = "torch-ucc",
    version = "0.1.0",
    ext_modules = [module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
