# 2022-2023 School Year Science Project

This repository contains code for my 2022 (9th grade) first semester science project. I have chosen to do a project on Machine Learning/Artificial Intelligence. The ultimate goal of this project to take the binary-search-based optimizer I derived for finding the minima of scalar valued functions (functions that take in a single number and output a single number) and extend it to multiple dimensions. Firstly linear regression and then neural networks. The hypothesis for the project originally was that it could get comparable results from less compute power, at least for neural networks.

## Installing and Running the Code

### Installing

To install the code for this project, it is recommended that you use git.
Install the code with git by running `git clone https://github.com/FrostyTheSouthernSnowman/2022-science-project` in the command prompt or cmd on windows or a bash/zsh terminal for MacOS/Linux.

### Running the Code

To run the code for this project, install the necessary dependencies with pip in a virtual environment by running `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt` in bash/zsh for MacOS/Linux or in git bash if you use windows (git bash is the preferred for windows users).
Once these dependencies are installed, run one if the files in the `code/completed/experiments` directory.

## Code overview

The extra markdown files in the root directory are just to keep my self organized. The code lives inside the `code` directory. Inside the code directory, there are the `work-in-progress` and `completed` directories. These directories contain work that is incomplete and complete, respectively. I will try to publish Medium articles about how the code works, what it does, and how I derived any formulas/algorithms.

## Contributing

Since this is a school project, I cannot accept any contributions until I have finished and turned in everything. After the seventh of November, I will start accepting
contributions.

## Contact

You can contact me via email (yosi_frost@icloud.com) if necessary.

# Graphs

The html for the graphs can be found on the pages.

- quadratic equation: [/code/completed/pages/graphs/quadratic.html](https://frostythesouthernsnowman.github.io/2022-science-project/graphs/quadratic.html)
- simple neural network: [/code/completed/pages/graphs/simple_neural_network.html](https://frostythesouthernsnowman.github.io/2022-science-project/graphs/simple_neural_network.html)
- simple neural network with modified binary search: [/code/completed/pages/graphs/simple_nn_modified_binary_search.html](https://frostythesouthernsnowman.github.io/2022-science-project/graphs/simple_nn_modified_binary_search.html)
- simple neural network with modified binary search after bug fix: [/code/completed/pages/graphs/simple_nn_modified_binary_search_and_bug_fix.html](https://frostythesouthernsnowman.github.io/2022-science-project/graphs/simple_nn_modified_binary_search_and_bug_fix.html)

## Experiments

- experiment-01: [/code/completed/experiments/experiment-01.py](/code/completed/experiments/experiment-01.py)
- experiment-02: [/code/completed/experiments/experiment-02.py](/code/completed/experiments/experiment-02.py)
- experiment-03: [/code/completed/experiments/experiment-03.py](/code/completed/experiments/experiment-03.py)
- experiment-04: [/code/completed/experiments/experiment-04.py](/code/completed/experiments/experiment-04.py)
- experiment-05: [/code/completed/experiments/experiment-05.py](/code/completed/experiments/experiment-05.py)
