# Import required functions
from setuptools import setup, find_packages


# Call setup function
setup(
    author="Florence d'Alché-Buc (Researcher), Luc Motte (Researcher), Awais Sani (Engineer), Danaël Schlewer-Becker (Engineer), Gaëtan Brison (Engineer)",
    description="Test package using IOKR method with the long term goal to develop a Structured-Prediction Package",
    name="IOKR",
    version="0.1.0",
    install_requires=["pandas","numpy","scipy","scikit-learn"],
    python_requires=">=2.7",
    ext_modules = cythonize(["_tree.pyx", "_splitter.pyx", "_criterion.pyx"]),
    language_level = "3"
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