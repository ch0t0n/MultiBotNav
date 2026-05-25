# Notes for CoRL 2026 Submission
This is the branch for running experiments from the KDD GPU servers.

## Conda environment activation
```
cd "/home/c/choton/reinforcement_learning/corl_2026/git/MultiBotNav/running_experiments" && source ~/.bashrc && conda activate rl4pag
```


## Currently running experiments 
### In GPU server: 
1. `/home/c/choton/reinforcement_learning/corl_2026/git/MultiBotNav/running_experiments/v1_wheeled.py`

### May 25, 2026
Today I developed the best environment for wheeled robots which now learns the optimal policy for every env variation. The code is in `running_experiments/v2_wheeled.py`.

### May 14, 2026
Today I have multiple updates:
1. First, I created two branches in the git repo `https://github.com/ch0t0n/MultiBotNav/`: `gpu_server` for experiments in KDD GPU server, and `experiment` for experiments/development in local machine(s).
2. In the `experiment` branch, I have updated the single file implementation `/single_file_implementation/example_training.py` so that it uses a single `JSON` configuration file for the wheeled robots instead of 10 different `.ini` files. Simulation using the new single configuration file is in `/simulation/wheeled_random_sim2.py`.
3. In the `experiment` branch, I have updated the `/writings/latex_draft1.tex` to match the writings from NeuRIPS 2026 (which is in `/writings/latex_precision_spraying.tex`). It still needs to be updated since the experiments for the wheeled robots are only using 3 robots instead of 2-5 robots. The `main` file in the CoRL 2026 overleaf draft (Link: https://www.overleaf.com/project/6a0362acb2390e693086c687) now has the writings from Cursor AI tool, and writings from Claude are in `Claude_test1.tex` and `Claude_test2.tex`. 
4. In the `gpu_server` branch, all previous exp versions are deleted (still on previous commits). The new `v1.py` is identical to the single file implementation `/single_file_implementation/example_training.py` in the `experiment` branch.


### May 12, 2026 
1. Since `v2_test.py` is not working, I created `v3_test.py` which uses another reward scaling from the prompt: `Reward scaling normalization for training stability`.


### May 11, 2026 
1. Based on the spraying environment developed for NeuRIPS, I created two environments: `v1_test.py` and `v2_test.py` where `v1_test.py` uses the existing reward model (from ICRA 2026 and IROS 2026 submission), and `v2_test.py` uses a scaled down version. Both scripts now have the updated environments for UAV and wheeled robot navigation.
2. The prompts for creating these environments are in the project `Multi-robot navigation in a stochastic environment` on **Claude**. The prompt name is `Updating path planning script with environmental uncertainties`. The reward scaling prompt is `Reward scaling normalization for training stability`.
3. From the results, `v2_test.py` is not working at all.