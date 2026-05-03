#! /usr/bin/env python

import os
import re
import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path
from sysconfig import get_paths
from tempfile import TemporaryDirectory

import requests
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, source_dir: Path):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])
        self.sourcedir = source_dir.resolve()


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension):
        pattern = re.compile(r".*lib.python\d.\d+$")
        for path in sys.path:
            if pattern.match(path):
                src_dir = path

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        boost_version = "1.91.0"
        onnxruntime_version = "1.25.0"

        boost_url = f"https://archives.boost.io/release/{boost_version}/source/boost_{boost_version.replace('.', '_')}.tar.gz"
        onnxruntime_url = f"https://github.com/microsoft/onnxruntime/releases/download/v{onnxruntime_version}/onnxruntime-linux-x64-{onnxruntime_version}.tgz"

        with TemporaryDirectory() as tmp_dir:

            for url in [boost_url, onnxruntime_url]:
                res = requests.get(url)
                with tarfile.open(fileobj=BytesIO(res.content)) as tar:
                    tar.extractall(tmp_dir)
                del res

            # example of cmake args
            config = "Debug" if self.debug else "Release"
            cmake_args = [
                "-DLIBRARY_OUTPUT_DIRECTORY=" + src_dir,
                "-DCMAKE_BUILD_TYPE=" + config,
                "-DPYTHON_INCLUDE_DIR=" + str(get_paths()["include"]),
                "-DPYTHON_EXECUTABLE=" + str(sys.executable),
                "-DCMAKE_INSTALL_PREFIX=" + str(build_temp),
                "-DBOOST_DIR="
                + str(
                    os.path.join(tmp_dir, f"boost_{boost_version.replace('.', '_')}")
                ),
                "-DONNXRUNTIME_DIR="
                + str(
                    os.path.join(
                        tmp_dir,
                        f"onnxruntime-linux-x64-{onnxruntime_version}",
                    )
                ),
            ]

            # example of build args
            build_args = ["--config", config, "--", "-j" + str(min(os.cpu_count(), 16))]

            subprocess.run(
                ["cmake", str(ext.sourcedir), *cmake_args], cwd=build_temp, check=True
            )
            subprocess.run(
                ["cmake", "--build", ".", "--target", "install", *build_args],
                cwd=build_temp,
                check=True,
            )


setup(
    name="qam",
    version="0.1",
    packages=find_packages(),
    ext_modules=[CMakeExtension("bolt", Path(__file__).parent)],
    cmdclass={
        "build_ext": CMakeBuild,
    },
)
