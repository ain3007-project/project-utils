from setuptools import setup, find_packages

setup(
    name='project_utils',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "torch",
        "huggingface_hub",
        "scikit-learn",
        "matplotlib",
        "numpy",
        "pandas",
        "pillow",
    ],
)
