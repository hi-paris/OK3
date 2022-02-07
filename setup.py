from setuptools import find_packages
# from setuptools import setup as setuppackage
from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy as np

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="OK3",
    version="0.0.1", 
    packages=find_packages(),
    author="???",
    description="???",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/hi-paris/OK3",
    #extensions = [Extension("*", ["*.pyx"])],
    #cmdclass={'build_ext': Cython.Build.build_ext},
    ext_modules=cythonize(["_tree.pyx", "_splitter.pyx", "_criterion.pyx"], language_level="3"),
    include_dirs=[np.get_include()],
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'cython',
    ],
    install_requires=[
        'numpy>=1.19.2',
        'cython',
    ],
    python_requires=">=2.7"
)

"""
import os

import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("tree", parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('m')
    config.add_extension("_tree",
                         sources=["_tree.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("_splitter",
                         sources=["_splitter.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    config.add_extension("_criterion",
                         sources=["_criterion.pyx"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])

    #config.add_subpackage("tests")

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
"""
