import setuptools

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="atm76",
    version="0.1.0",
    author="Steven H. Berguin",
    author_email="stevenberguin@gmail.com",
    description="Differentiable 1976 Atmosphere",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shb84/ATM76.git",
    packages=setuptools.find_packages(),
    package_data={},
    install_requires=["numpy>=1.16", "genn"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
