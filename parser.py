import argparse

parser = argparse.ArgumentParser(description="Every RL algorithm with Cartpole env")
parser.add_argument('--env', type=str, default='CartPole-v1',
                     help='cartpole env')
parser.add_argument('--algo',type=str,default='dqn',
                    help='select algorithm')
parser.add_argument('--phase', type=str, default='train',
                    help='train or test')
parser.add_argument('--render', action='store_true', default=True,
                    help='render or not')
parser.add_argument('--load', type=str, default=None,
                    help='load model')
parser.add_argument('--seed',type=int, default=0,
                    help='set seed number')
parser.add_argument('--iterations', type=int, default=500,
                     help='iterations to run and train agent')
parser.add_argument('--eval_per_train',type=int, default=50,
                    help='evaluation number per training')
parser.add_argument('--max_step', type=int, default=500,
                    help='max episode step')
parser.add_argument('--threshold_return', type=int, default=500,
                    help='solved requirement for success in given environment')
parser.add_argument('--tensorboard', action='store_true', default=True)
parser.add_argument('--gpu_index', type=int, default=0)
args = parser.parse_args()
