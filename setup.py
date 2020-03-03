import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tiger-interpreter-gmeccles@gmail.com",
    version="0.0.1",
    author="Gordon Eccles",
    author_email="gmeccles@gmail.com",
    description=(
        "A Python implementation of an interpreter for the Tiger language."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gordoneccles/tiger",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
