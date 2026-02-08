"""Build configuration for the predted C extension.

The C extension provides a fast implementation of the 36 structural
features used by predTED.  It is optional — the pure-Python fallback
in predted/features.py is used when the extension is not available.
"""

import os
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Allow the C extension to fail gracefully."""

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            print(f"WARNING: Could not build C extension: {e}")
            print("Falling back to pure-Python feature computation.")


ext = Extension(
    "predted._features_c",
    sources=[
        "c_src/predted_features.c",
        "c_src/_features_module.c",
    ],
    include_dirs=["c_src"],
    extra_compile_args=["-O2"],
)

setup(
    ext_modules=[ext],
    cmdclass={"build_ext": OptionalBuildExt},
)
