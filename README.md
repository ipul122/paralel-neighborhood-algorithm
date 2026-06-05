# Parallel Neighborhood Algorithm (NA) for 1D Magnetotelluric (MT) Data Inversion & Uncertainty Analysis

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="Images/curve_5layer.png" width="345">
  <img src="Images/model_5layer.png" width="300">
</p>

<p align="center">
  <img src="Images/MT 1D Inversion (5 Layer).png" width="650">
</p>

This repository contains a high-performance Python implementation of the *Neighborhood Algorithm* (Sambridge, 1999). It is optimized using multi-core parallelization (`joblib`) and Just-In-Time compilation (`Numba JIT`) for 1D Magnetotelluric (MT) data inversion and uncertainty quantification.

Core algorithm framework adapted from: `https://github.com/auggiemarignier/neighpy`

## 🚀 Key Features

- **Fast Search Stage (Numba JIT):** Accelerates the MT forward modeling function using the `@njit(parallel=True, fastmath=True)` decorator for high-speed machine compilation.
- **Multi-Core Walkers:** Distributes the random walk sampling computation across all logical CPU processors using the `joblib` parallel architecture.

## 📦 System Requirements & Installation

### Prerequisites
- Python 3.9 or Python 3.10

### Installation Steps

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/ipul122/paralel-neighborhood-algorithm.git](https://github.com/ipul122/paralel-neighborhood-algorithm.git
   cd paralel-neighborhood-algorithm
2. Install all required dependencies via the requirements.txt file:
   ```bash
   pip install -r requirements.txt
3. To run the Neighborhood Algorithm (Global Inversion & Uncertainty Analysis):
   ```bash
   python main.py
4. To run the Levenberg-Marquardt Inversion (Gradient-based Local Inversion):
   ```bash
   cd LM
   python main.py
   
### Field Data File Setup
Before running the inversion, you must format your observed field data and save it as `MT/ {CUSTOM_NA.txt or CUSTOM_LM.txt}`. The file must use a tab-separated (`\t`) delimiter and contain a header row. 

The columns must strictly follow this order:
1. **Frequency** (in Hz)
2. **Apparent Resistivity** (in $\Omega\cdot\text{m}$)
3. **Phase** (in degrees)
