# setup for cython functions
# Whitebeam | Kieran Molloy | Lancaster University 2020

from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

NAME = "whitebeam"
VERSION = "0.0.1"
DESCR = "Whitebeam is a framework for creating decision tree functions."
REQUIRES = ["numpy", "cython", "joblib"]

AUTHOR = "Kieran Molloy"
EMAIL = "k.molloy@lancaster.ac.uk"

SRC_DIR = "whitebeam"
PACKAGES = [SRC_DIR]

ext_1 = Extension(SRC_DIR + ".core._whitebeam",
                  sources=[SRC_DIR + "/core/_whitebeam.pyx"],
                  libraries=[],
                  include_dirs=[np.get_include()])

if __name__ == "__main__":
    setup(ext_modules = [ext_1],
        cmdclass={"build_ext": build_ext},
        install_requires=REQUIRES,
        packages=PACKAGES,
        zip_safe=False,
        name=NAME,
        version=VERSION,
        description=DESCR,
        author=AUTHOR,
        author_email=EMAIL
        )