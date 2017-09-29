from os import path
from codecs import open
from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))


with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name = "trickster",
    version = "0.0.0",
    author = "Bogdan Kulynych",
    author_email = "hello@bogdankulynych.me",
    description = "Generate adversarial examples for discrete and mixed feature domains",
    url = "https://github.com/bogdan-kulynych/constable",
    license = "MIT",
    packages=find_packages(exclude=["tests", "notebooks"]),
    long_description=long_description,
    install_requires=[
        "keras",
        "tensorflow",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
