
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="PassivePy",
    version="0.2.19",
    author="Mitra Mirshafiee",
    author_email="mitra.mirshafiee@gmail.com",
    description="A package for processing large corpora and detecting passive voice.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mitramir55/PassivePy",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3', 

        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    keywords='passive voice, text analysis, spacy, dependency parsing, part of speech tagging',
    package_dir={'': 'PassivePyCode'},
    packages=find_packages('PassivePyCode'),  
    python_requires=">=3.6",

    install_requires=['termcolor', 'tqdm', 'spacy>=3.0.0'],


)