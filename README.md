# Monte Carlo Tree Search using Network-based Implicit Minimax Backups
This repository contains an implementation of Monte Carlo Tree Search using Network-based Implicit Minimax Backups (MCTS<sub>NIM</sub>) based on the code used for my thesis.

## Overview
MCTS<sub>NIM</sub> is a variation of MCTS that uses implicit minimax backups in combination with a neural network to estimate the value of different game states. The neural networks used by the algorithm are trained using the [descent framework](https://arxiv.org/abs/2008.01188). The implementation is written in Java using [Ludii](https://ludii.games/) and the [DeepLearning4J library](https://deeplearning4j.konduit.ai/). Some minor modifications have been made to a fork of Ludii, which can be found [here](https://github.com/GitHubByJelle/Ludii).

<p align="center" width="100%">
    <img src="images/mcts-nim.gif" alt="Visualisation of MCTS_NIM playing a game of Breakthrough against Descent using Ludii's heuristic function" width="70%">
</p>

## Installation
All used dependencies can be installed using the `pom.xml` file. Additionally, the modified Ludii can be installed by using the `Ludii-1.3.6-Thesis.jar`.

## Implementation of MCTS<sub>NIM</sub>
The implementation of MCTS<sub>NIM</sub> can be found in the four variants `MCTS_base`, `MCTS_alpha`, `MCTS_exploration`, and `MCTS_combine`. These variants utilize the modified `MCTS` agent from Ludii. All building blocks used by MCTS<sub>NIM</sub> can be found in `MCTSStrategies`, which can easily be used by Ludii's `MCTS` agent.

## Usage
The implementation provides both training and testing of the implemented agents.

### Training
Training can be done using the `Training.LearningManager`, which uses the descent framework. Training can be started by running the `Training.Learning` file. If no configuration is used for these experiments, the template configuration file (`configurations/templateTraining.properties`) will be used. This template can also be adjusted to train with other configurations.

### Testing
The search algorithms can be tested by running `Experiments.Playground`, which also uses the template configuration file (`configurations/template.proporties`) if no configuration file is used. Also this template can be adjusted to test with other configurations.

