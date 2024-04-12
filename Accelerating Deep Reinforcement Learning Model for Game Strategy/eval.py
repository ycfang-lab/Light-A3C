import os
from env import Env
import argparse
import torch
import random
from datetime import datetime
from agents.mobile_agent import Agent

parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='defender', help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(15e6), metavar='STEPS',
                    help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH',
                    help='Max episode length (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                    help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β',
                    help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(32e3), metavar='τ',
                    help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--learn-start', type=int, default=int(80e3), metavar='STEPS',
                    help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS',
                    help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N',
                    help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N',
                    help='Number of transitions to use for validating Q')
parser.add_argument('--log-interval', type=int, default=25000, metavar='STEPS',
                    help='Number of training steps between logging status')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')


def log(s, path):
    with open(os.path.join(path, "time.txt"), "a+")as f:
        f.write('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s + "\n")


def eval(args):
    env = Env(args)
    env.eval()
    dqn = Agent(args, env)
    # action_space = env.action_space()
    state = torch.load("./results/defender/mobile_cnn_liner/model.pth", 'cpu')
    dqn.online_net.load_state_dict(state)
    dqn.eval()
    # dqn.binary_online.binarization()
    print("loaded model")
    done = True
    T = 0
    reward_sum = 0
    maxnum = -1e5
    log("start", "./")
    while True:
        while True:
            if done:
                state, reward_sum, done = env.reset(), 0, False
            action = dqn.act_e_greedy(state)  # random.randint(0, action_space - 1) Choose an action ε-greedily
            state, reward, done = env.step(action)  # Step
            reward_sum += reward
            T += 1
            env.render()
            if done or T >= 100000:
                break
        if reward_sum > maxnum:
            maxnum = reward_sum
        if T >= 1000000:
            break
        print(T, maxnum)
    env.close()
    log("end", "./")


if __name__ == '__main__':
    args = parser.parse_args()
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))
    random.seed(args.seed)
    torch.manual_seed(random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(random.randint(1, 10000))
        torch.backends.cudnn.enabled = False
    else:
        args.device = torch.device('cpu')
    eval(args)
