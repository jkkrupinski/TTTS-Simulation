# TTTS-Simulation

**Project Description:**
This project is a part of a Master Thesis. Main goal is to create a simulation for a robotic arm to perform TTTS surgery. Robot will be trained using reinforcement learning on a custom created environment.

![Project Logo](https://example.com/logo.png) 

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Features

- List of features and functionalities.
- Highlight any unique aspects or key benefits.

## Installation

### Prerequisites

- **Python version: 3.9**
- **Other dependencies:**
    - numpy, 
    - mujoco, 
    - stable_baselines3, 
    - gymnasium, 
    - opencv, 
    - ikpy,
    - imageio

### Steps

1. Clone the repository:

    ```sh
    git clone https://github.com/jkkrupinski/TTTS-Simulation.git
    cd TTTS_SIMULATION
    ```

2. Create a Conda environment (optional but recommended):

    ```sh
    conda env create -f environment.yml
    conda activate my_env
    ```

3. Install dependencies:

    ```sh
    pip install -r requirements.txt
    ```


## Usage

### Basic Usage


- To run environment with random actions run environment.py script

    ```sh
    python environment.py
    ```

- To run a test of the trained model on the simulation environment run test.py script

    ```sh
    python test.py
    ```

