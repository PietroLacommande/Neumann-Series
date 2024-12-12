# Simplifying Inverse Matrix Calculation with the Neumann Series

This repository provides an implementation of the Neumann series to approximate the inverse of a matrix.
The Neumann series method is particularly useful for matrices that are close to the identity matrix, offering a computationally efficient alternative to direct inversion.

## Overview

Matrix inversion can be computationally expensive, especially for large matrices.
The Neumann series offers a way to approximate the inverse for matrices \( A \) where \( \|I - A\| \) is small. This repository includes:
- An explanation of the Neumann series for matrix inversion.
- An implementation in CUDA.
- MATLAB code for generating a matrix A, calculating the follow up matrixes for Neumann series and calculating the inverse of matrix A

## Installation

Clone the repository:
git clone https://github.com/yourusername/neumann-inverse-matrix.git

## Licence

This project is licensed under the MIT License

## Responsibility and Code Release Practices

When releasing source code on GitHub, it is essential to follow responsible practices.
- Pull requests must be validated by at least one team member
- Unit tests should be ran automatically in a CI/CD pipeline
- Syntaxic tests with tools like SonarCloud should also be ran to validate the code
