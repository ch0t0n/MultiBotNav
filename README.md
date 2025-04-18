# Optimal Multi-Robot Path Planning For Herbicide Spraying Using Reinforcement Learning

This is the codebase for the ICRA 2025 paper "Optimal Multi-Robot Path Planning For Herbicide Spraying Using Reinforcement Learning", written by Jahid Chowdhury Choton, John Woods, Raja Farrukh Ali, and William Hsu. In this paper, we present a Reinforcement Learning (RL) solution for multi-robot systems used for spraying herbicide. Our contributions include:

* Developing a novel, customizable RL environment that represents an agricultural field with 3 spraying robots
* Analyzing 4 state-of-the-art RL algorithms across 10 different environments
* Creating a simultion framework of the environment using the CoppeliaSim robot simulator

## Setup

It is recommended to run this codebase on Linux. The necessary packages and libraries needed to run the code are provided in the `environment.yaml` conda file. If you do not have conda installed on your machine, download it [here](https://docs.anaconda.com/miniconda/miniconda-install/). Once it is installed, run the following command to set up the environment:

```
conda env create -f environment.yaml
```

If you update the environment by installing or removing packages, please update the conda file with the following command:

```
conda env export --no-builds > environment.yaml
```

## Generating Experiments

10 sets of experiments are already provided in this repository, with sets 1-8 having a field size of 50x50 and sets 9-10 having a field size of 100x100. To generate a single 50x50 field size experiment, simply run:

```
python3 generate_experiments.py
```

Newly generated experiments will be placed in the `experiments` directory. Additional experiments with different sizes can be generated using this command format:

```
python3 generate_experiments.py [number of experiments] --max_size [field size]
```

## Training

### On your local machine

To train an algorithm with the default configuration, run the following command:

```
python3 train.py --algorithm A2C --set 1
```

The currently implemented algorithms are `A2C`, `PPO`, `TRPO`, and `RecurrentPPO`. The possible values for --`set` depend on the number of sets in the `experiments` directory. Training can be further configured using the following command format:

```
python3 train.py --algorithm {A2C, PPO, TRPO, RecurrentPPO} --set [set number] --verbose {0 for no output, 1 for info, 2 for debug} --gamma [discount factor] --steps [number of training steps] --num_envs [number of parallel environments] --resume {True for resuming training, False for new model}
```

### On Compute Clusters

Slurm scripts for training the model are also provided in the `training_scripts` directory. To run all experiments, use the command:

```
sbatch training_scripts/train_all.sh
```

To run just a single experiment, first configure the experiment in the `training_scripts/train_one.sh` file. Then, run the following command to start it:

```
sbatch training_scripts/train_one.sh
```

### Viewing Results

Training for 1 million timesteps takes around 1-4 hours, depending on the algorithm. Once complete, the trained model will be saved in the `trained_models` directory.

Logs containg the reward info are also generated as the model trains. To view them, simply run:

```
tensorboard --logdir=./logs
```

## Simulation

**Note:** Simulation can only be done once you have fully trained a model.

### PyGame

To simulate a trained model in PyGame, run the following command:

```
python3 run.py --algorithm A2C --set 1 --simulate False
```

### CoppeliaSim

First, you will need to download the CoppeliaSim robotics simulator [here](https://coppeliarobotics.com/). Once it is installed, open the `simulation_env/drone_test_scene_aug14.ttt` scene in the simulator.

**IMPORTANT:** You will need to reopen the scene each time before running the simulation. Never save changes to the scene file when closing.

To simulate a trained model in CoppeliaSim, run the following command:

```
python3 run.py --algorithm A2C --set 1 --simulate True
```

## Plotting

We also provide some scripts to aid with plotting results. To plot the layouts of the provided 10 experiment sets, run the following command:

```
python3 plotting/plot_fields.py
```

Once the experiments have been run, you can plot individual experiment performance and average algorithm performance with the following command:

```
python3 plotting/plot_results.py
```
