"""
Collection of NN feature extractors for training agents

We took model implementations from
https://github.com/mila-iqia/babyai/blob/dyth-babyai-v1.1/babyai/model.py

BaybAIFullyObsCNN won't use
  * English instructions
  * memory (rnn)
  * FiLM, so they are removed from the code.

It may not be the best architecture for our use case, but
we will retain this architecture consistently over experiments.
"""
import gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from parl_agents.nn_models.utils import initialize_parameters

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor
)


class ImageBOWEmbedding(nn.Module):
    """
    max_value takes observation image dimension, and
    embedding_dim is feature dimension, hard coded to 128
    """
    def __init__(self, max_value=147, embedding_dim=128):
        super().__init__()
        self.max_value = max_value
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(3 * max_value, embedding_dim)
        self.apply(initialize_parameters)

    def forward(self, inputs):
        self.embedding = self.embedding.to(inputs.device)
        offsets = th.Tensor([0, self.max_value, 2 * self.max_value]).to(inputs.device)
        inputs = (inputs + offsets[None, :, None, None]).long()
        return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)


class BabyAIFullyObsCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)

        # the following parameters are the same parameters as shown in
        # https://github.com/mila-iqia/babyai/blob/dyth-v1.1-and-baselines/babyai/model.py
        # except for the additional nn.ReLU(), nn.Flatten() and nn.Linear() after nn.MaxPool2d
        #
        # cnn output = (Width + 2*padding - Kernel + 1)//Stride + 1
        # (N, 3, row==H, col==W)
        # where n is the number of batches,
        # 3 is the symbolic features or the number of channels,
        # H and W are 15, 22, 29 for 2, 3, 4 rooms with size 8
        # why max_value is 147? See this in BabyAI format.py
        # class ObssPreprocessor:
        #     def __init__(self, model_name, obs_space=None, load_vocab_from=None):
        #         self.image_preproc = RawImagePreprocessor()
        #         self.instr_preproc = InstructionsPreprocessor(model_name, load_vocab_from)
        #         self.vocab = self.instr_preproc.vocab
        #         self.obs_space = {
        #             "image": 147,
        #             "instr": self.vocab.max_size
        #         }

        self.max_value = 147
        self.embedding = nn.Embedding(3 * self.max_value, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Flatten()
        )

        # non-pixel
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),  # W //2
        #     nn.Flatten()
        # )

        # pixel
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=features_dim, kernel_size=(8, 8), stride=(8, 8), padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(2, 2), stride=(1, 1), padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0), 
        #     nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0), 
        #     nn.Flatten()
        # )

        with th.no_grad():
            x = th.as_tensor(observation_space.sample()[None]).float()
            x = x / 255
            print(x.shape)
            offsets = th.Tensor([0, self.max_value, 2 * self.max_value])
            print(offsets.shape)
            x = (x + offsets[None, :, None, None]).long()
            print(x.shape)
            x =  self.embedding(x).sum(1).permute(0, 3, 1, 2)
            x = self.cnn(x)
            n_flatten = x.shape[1]
            print("n_flatten:{}".format(n_flatten))

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

        self.apply(initialize_parameters)       # Initialize parameters correctly

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # from (batch, row, col, channel) to (batch, channel, row, col)
        # x = th.transpose(th.transpose(observations, 1, 3), 2, 3); done by BaseAlgorithm
        # x = observations.float()
        
        offsets = th.Tensor([0, self.max_value, 2 * self.max_value]).to(observations.device)
        x = (observations + offsets[None, :, None, None]).long()
        x =  self.embedding(x).sum(1).permute(0, 3, 1, 2)
        
        x = self.cnn(x)
        x = self.linear(x)

        # if adding them in seq.
        # x = F.relu(x)
        # x = x.reshape(x.shape[0], -1)       # flatten        [Nbatch, FeatureDim]
        return x



class BabyAIFullyObsCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)

        # the following parameters are the same parameters as shown in
        # https://github.com/mila-iqia/babyai/blob/dyth-v1.1-and-baselines/babyai/model.py
        # except for the additional nn.ReLU(), nn.Flatten() and nn.Linear() after nn.MaxPool2d
        #
        # cnn output = (Width + 2*padding - Kernel + 1)//Stride + 1
        # (N, 3, row==H, col==W)
        # where n is the number of batches,
        # 3 is the symbolic features or the number of channels,
        # H and W are 15, 22, 29 for 2, 3, 4 rooms with size 8
        # why max_value is 147? See this in BabyAI format.py
        # class ObssPreprocessor:
        #     def __init__(self, model_name, obs_space=None, load_vocab_from=None):
        #         self.image_preproc = RawImagePreprocessor()
        #         self.instr_preproc = InstructionsPreprocessor(model_name, load_vocab_from)
        #         self.vocab = self.instr_preproc.vocab
        #         self.obs_space = {
        #             "image": 147,
        #             "instr": self.vocab.max_size
        #         }

        self.max_value = 147
        self.embedding = nn.Embedding(3 * self.max_value, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Flatten()
        )

        # non-pixel
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),  # W //2
        #     nn.Flatten()
        # )

        # pixel
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=features_dim, kernel_size=(8, 8), stride=(8, 8), padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(2, 2), stride=(1, 1), padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0), 
        #     nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0), 
        #     nn.Flatten()
        # )

        with th.no_grad():
            x = th.as_tensor(observation_space.sample()[None]).float()
            x = x / 255
            print(x.shape)
            offsets = th.Tensor([0, self.max_value, 2 * self.max_value])
            print(offsets.shape)
            x = (x + offsets[None, :, None, None]).long()
            print(x.shape)
            x =  self.embedding(x).sum(1).permute(0, 3, 1, 2)
            x = self.cnn(x)
            n_flatten = x.shape[1]
            print("n_flatten:{}".format(n_flatten))

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

        self.apply(initialize_parameters)       # Initialize parameters correctly

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # from (batch, row, col, channel) to (batch, channel, row, col)
        # x = th.transpose(th.transpose(observations, 1, 3), 2, 3); done by BaseAlgorithm
        # x = observations.float()
        
        offsets = th.Tensor([0, self.max_value, 2 * self.max_value]).to(observations.device)
        x = (observations + offsets[None, :, None, None]).long()
        x =  self.embedding(x).sum(1).permute(0, 3, 1, 2)
        
        x = self.cnn(x)
        x = self.linear(x)

        # if adding them in seq.
        # x = F.relu(x)
        # x = x.reshape(x.shape[0], -1)       # flatten        [Nbatch, FeatureDim]
        return x


class BabyAIFullyObsSmallCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)

        self.max_value = 147
        self.embedding = nn.Embedding(3 * self.max_value, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            # nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            # nn.BatchNorm2d(features_dim),
            # nn.ReLU(),
            nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Flatten()
        )


        with th.no_grad():
            x = th.as_tensor(observation_space.sample()[None]).float()
            x = x / 255
            print(x.shape)
            offsets = th.Tensor([0, self.max_value, 2 * self.max_value])
            print(offsets.shape)
            x = (x + offsets[None, :, None, None]).long()
            print(x.shape)
            x =  self.embedding(x).sum(1).permute(0, 3, 1, 2)
            x = self.cnn(x)
            n_flatten = x.shape[1]
            print("n_flatten:{}".format(n_flatten))

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

        self.apply(initialize_parameters)       # Initialize parameters correctly

    def forward(self, observations: th.Tensor) -> th.Tensor:
        offsets = th.Tensor([0, self.max_value, 2 * self.max_value]).to(observations.device)
        x = (observations + offsets[None, :, None, None]).long()
        x =  self.embedding(x).sum(1).permute(0, 3, 1, 2)
        
        x = self.cnn(x)
        x = self.linear(x)
        return x


class BabyAIFullyObsSmallCNNDict(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        image_observation_space = observation_space.spaces['image']

        self.max_value = 147
        self.embedding = nn.Embedding(3 * self.max_value, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2),
                      padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            # nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            # nn.BatchNorm2d(features_dim),
            # nn.ReLU(),
            nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2),
                      padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Flatten()
        )

        with th.no_grad():
            x = th.as_tensor(image_observation_space.sample()[None]).float()
            x = x / 255
            print(x.shape)
            offsets = th.Tensor([0, self.max_value, 2 * self.max_value])
            print(offsets.shape)
            x = (x + offsets[None, :, None, None]).long()
            print(x.shape)
            x = self.embedding(x).sum(1).permute(0, 3, 1, 2)
            x = self.cnn(x)
            n_flatten = x.shape[1]
            print("n_flatten:{}".format(n_flatten))

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        label_observation_space = observation_space.spaces['label']
        # self.label_embedding = nn.Embedding(num_embeddings=label_observation_space.n, embedding_dim=features_dim)
        self.label_embedding = nn.Linear(label_observation_space.n, features_dim)       # sb3 coverts to onehot!

        self.linear2 = nn.Sequential(
            nn.Linear(features_dim * 2, features_dim),
            nn.ReLU()
        )
        self.apply(initialize_parameters)  # Initialize parameters correctly

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations['image']
        offsets = th.Tensor([0, self.max_value, 2 * self.max_value]).to(x.device)
        x = (x + offsets[None, :, None, None]).long()
        x = self.embedding(x).sum(1).permute(0, 3, 1, 2)
        x = self.cnn(x)
        x = self.linear(x)

        y = observations['label']       # B x num_env x dict_size       # assume env size 1 squeeze num_env
        y = th.squeeze(y)
        y = self.label_embedding(y)
        # in case of prediction, batch size is 1 that 1 x 1 x dict_size became dict_size
        if y.ndim == 1:
            y = y.reshape((1, -1))
        z = th.cat((x,y), dim=1)
        z = self.linear2(z)
        return z

