"""Install package."""
from setuptools import setup, find_packages

setup(
    name="HiProtoNet",
    version="0.0.1",
    description=("Interpretable detection of Aortic Stenosis Severity with Hierarchical Prototypical Neural Network in Hyperbolic Space"),
    long_description=open("README.md").read(),
    url="https://github.com/DeepRCL/HiProtoNet",
    install_requires=["numpy"],
    packages=find_packages("."),
    zip_safe=False,
)
