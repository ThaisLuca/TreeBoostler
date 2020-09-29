---

# TreeBoostler

> **TreeBoostler** is a transfer learning method that transfer statistical relational models learned through gradient-boosting approach.  

*boostsrl* is a Python package with wrappers for creating background knowledge and performing learning and inference. We used a modification of the original boostsrl package to work with a BoostSRL modification that allows the algorithm to learn parameters and perform refinement on given trees, as well transfer learning. It runs the transfer learning/theory revision modification of the BoostSRL developed in the following repositories:

https://github.com/rodrigoazs/BoostSRL
https://github.com/rodrigoazs/boostsrl-python-package

### Modified to perform Transfer Learning/Theory Revision

## Getting Started

### Prerequisites

* Java 1.8
* Python (2.7, 3.3, 3.4, 3.5, 3.6)
* subprocess32 (if using Python 2.7: `pip install subprocess32`)
* graphviz-0.8

### Basic Usage

* Run the experiments on transfer_experiment.py or learning_curve.py
