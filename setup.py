# setup.py
from setuptools import setup, find_packages

setup(
    name="sentiment_analysis_api",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "pandas",
        "numpy",
        "scikit-learn",
        "mlflow",
    ],
)
