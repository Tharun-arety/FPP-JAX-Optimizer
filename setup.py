from setuptools import find_packages, setup


setup(
    name="fpp-jax-optimizer",
    version="0.1.0",
    description="Differentiable Fiber Patch Placement optimizer for Type IV COPV dome workflows",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
