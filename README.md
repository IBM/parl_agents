# parl_agents
This repository offers pytorch based Hierarchical RL agents extending stable-baselines3

## Related Repos
we use `parl_agent`, `parl_benchmark`, `parl_annotations` together.
* [parl_agents](https://github.com/IBM/parl_agents): Hierarchical RL agent codes 
* [parl_minigrid](https://github.com/IBM/parl_minigrid): Add-on to the minigrid environemtns
* adding different kinds of annotation to RL task, we extend `parl_annotations`
* adding new annotated RL environments, we addd new `parl_benchmark` such as `parl_minigrid`

# Install
* first create a conda environment for installing parl_annotations, parl_agents, parl_minigrid.
```
$ conda create -n parl python=3.7
``` 
* install packages as editable library
```
pip install -r requirements
pip install -e .
```

## Stable Baselines3
* To avoid stable-baselines version issues, this repo stores src from
```
pip install -e git+https://github.com/DLR-RM/stable-baselines3.git@v1.5.0#egg=stable_baselines3
```
* We can directly install `stable_baselines3` using setup.py
```
$ cd src; pip install -e .
```

# Usage
There are sample scripts for running `hppo`, `ppo`, and `dqn` agents under
```test_scripts```.


# Citations
* 2021 ICAPS PRL Workshop paper
```
@inproceedings{lee2021ai,
  title={AI Planning Annotation in Reinforcement Learning: Options and Beyond},
  author={Lee, Junkyu and Katz, Michael and Agravante, Don Joven and Liu, Miao and Klinger, Tim and Campbell, Murray and Sohrabi, Shirin and Tesauro, Gerald},
  booktitle={Planning and Reinforcement Learning PRL Workshop at ICAPS},
  year={2021}
}
```

* 2023 NEURIPS GenPlan Workshop paper
```
@inproceedings{lee2021ai,
  title={Hierarchical Reinforcement Learning with AI Planning Models},
  author={Lee, Junkyu and Katz, Michael and Agravante, Don Joven and Liu, Miao and Tasse, Geraud Nangue and Klinger, Tim and Sohrabi, Shirin},
  booktitle={Generalization in Planning GenPlan Workshop at NEURIPS},
  year={2023}
}
```