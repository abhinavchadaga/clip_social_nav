import os
import glob
from termcolor import cprint
import argparse


parser = argparse.ArgumentParser(
    description='process all rosbags to completion')
parser.add_argument('--save_data_path',
                    '-d',
                    dest="save_data_path",
                    default='/home/abhinavchadaga/CS/clip_social_nav/data')
args = parser.parse_args()

# clear unprocessed files
unprocessed = glob.glob(os.path.join(args.save_data_path, '*_data*'))
for r in unprocessed:
    cprint(f'removed {r}', 'red')
    os.system(f'rm -r {r}')

cprint('removed unprocessed data!', 'red', attrs=['bold'])
