import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="affine_tf",
    version="0.0.1",
    author="Alexander Barmin",
    author_email="barmin1@mail.ru",
    description="Affine transform implementation",
    url="https://github.com/Alek-dr/Affine_tf",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language : Python 3.7",
        "License : MIT License",
        "Operating System : OS Independent",
    ],
    python_requires='>=3.7',
)