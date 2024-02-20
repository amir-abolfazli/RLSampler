import argparse

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='parameters')

        self.parser.add_argument("--device", default="cuda", type=str, choices=["cpu", "cuda"])
        ############## Env and dataset related #############
        self.parser.add_argument("--env", default="csgym-v1")  # OpenAI gym-based configuration sampling enviornment
        self.parser.add_argument("--train_seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
        self.parser.add_argument("--eval_seed", default=10, type=int)  # Sets Gym, PyTorch and Numpy seeds
        self.parser.add_argument("--t", default=2, type=int, choices=[2,3])  # parameter t for t-wise feature combinations
        self.parser.add_argument("--fm", default=8, type=int)  # id for the feature model
        self.parser.add_argument("--train_n_episodes", default=1000, type=int)  # number of episodes for training
        self.parser.add_argument("--eval_n_episodes", default=100, type=int)  # number of episodes for evaluating
        self.parser.add_argument("--train_max_steps", default=int(1e4),type=int)  # Max time steps to run environment for training the agent
        self.parser.add_argument("--eval_max_steps", default=500,type=int)  # Max time steps to run environment  for evaluatig the agent

        self.args = self.parser.parse_args()

param = Param()
args = param.args
