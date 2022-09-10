# parl_agents
* pytorch based implementation of parl agents by extending stable-baselines3


## Install
* we will use `parl_agent`, `parl_benchmark`, `parl_annotation` together.

```
conda create -n parl_minigrid python=3.7
conda activate parl_minigrid
pip install -r requirements.txt
pip install -e .
```
* install `parl_annotaions` and `parl_minigrid` in the same conda environment

# Citations
```
@inproceedings{lee2021ai,
  title={AI Planning Annotation in Reinforcement Learning: Options and Beyond},
  author={Lee, Junkyu and Katz, Michael and Agravante, Don Joven and Liu, Miao and Klinger, Tim and Campbell, Murray and Sohrabi, Shirin and Tesauro, Gerald},
  booktitle={Planning and Reinforcement Learning PRL Workshop at ICAPS},
  year={2021}
}
```

# License
Apache-2.0 License


# Referencces
This project utilizes the following opensource projects.
```
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}

@misc{gym_minigrid,
  author = {Chevalier-Boisvert, Maxime and Willems, Lucas and Pal, Suman},
  title = {Minimalistic Gridworld Environment for OpenAI Gym},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/maximecb/gym-minigrid}},
}

@misc{hui2020babyai,
    title={BabyAI 1.1},
    author={David Yu-Tung Hui and Maxime Chevalier-Boisvert and Dzmitry Bahdanau and Yoshua Bengio},
    year={2020},
    eprint={2007.12770},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}

@Misc{alkhazraji-et-al-zenodo2020,
  author =       "Yusra Alkhazraji and Matthias Frorath and Markus Gr{\"u}tzner
                  and Malte Helmert and Thomas Liebetraut and Robert Mattm{\"u}ller
                  and Manuela Ortlieb and Jendrik Seipp and Tobias Springenberg and
                  Philip Stahl and Jan W{\"u}lfing",
  title =        "Pyperplan",
  publisher =    "Zenodo",
  year =         "2020",
  doi =          "10.5281/zenodo.3700819",
  url =          "https://doi.org/10.5281/zenodo.3700819",
  howpublished = "\url{https://doi.org/10.5281/zenodo.3700819}"
}

@article{helmert2006fast,
  title={The Fast Downward planning system.},
  author={Helmert, Malte},
  journal={Journal of Artificial Intelligence Research},
  volume={26},
  pages={191--246},
  year={2006}
}

```
