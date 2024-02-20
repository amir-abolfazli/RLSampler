import numpy as np
import torch
import gym
import csgym
from rl_agents.bdq import BDQ
from params import args
from tqdm import tqdm


def main(args):

    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')

    if (torch.cuda.is_available()):
        device = torch.device('cuda:0')
        # device = torch.device('cuda')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    print("============================================================================================")


    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)



    feature_models = {
        1: "BerkeleyDBC",
        2: "Dune",
        3: "JavaGC",
        4: "JHipster",
        5: "lrzip",
        6: "Polly",
        7: "VP9",
        8: "X264",
    }

    fm_name = feature_models[args.fm]

    env = gym.make(args.env, fm_name=fm_name, t=args.t, seed=args.train_seed, render_mode='human', rl_model_name="RLSampler", mode='train')

    nS = env.observation_space.shape[0]
    nA = (2, env.n_actions)

    agent = BDQ(nS,
                nA,
                epsilon_decay_steps=10000,
                new_actions_prob=0.1,
                buffer_size_max=30000,
                buffer_size_min=100,
                beta_increase_steps=30000,
                device=device)

    episode = 0

    for i in range(1, args.train_n_episodes + 1):
        episode += 1
        episode_reward = 0.
        state, done = env.reset(), False

        for j in tqdm(range(1, args.train_max_steps + 1)):

            discrete_actions = agent.act(state)

            actions = discrete_actions.astype(np.float32)

            next_state, reward, done, info = env.step(actions)

            terminal = done
            terminal = 1. if terminal else 0.0

            agent.experience(state, discrete_actions, reward, next_state, terminal)
            agent.train()

            state = next_state
            episode_reward += reward

            if done or j % args.train_max_steps == 0:
                break

    agent.save_net(env.get_model_path())


if __name__ == '__main__':
    main(args)