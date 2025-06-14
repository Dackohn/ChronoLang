from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension module
extensions = [
    Extension(
        "interpreter",  # Module name
        ["interpreter.py"],  # Source files
        include_dirs=[numpy.get_include()],  # Include NumPy headers
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )
]

setup(
    name="ChronoLang Interpreter",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': 3}),
    zip_safe=False,
)