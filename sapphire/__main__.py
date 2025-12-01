"""
This file is run automatically if the user does "python -m sapphire <parameters.JSON> on the command line
NOTE: *must* include the -m flag in the python call so it runs as a python package and explicit relative imports work.
"""

from . import run # imports sapphire.run() driver from __init__.py 
import argparse



def main():

    print('welcome to sapphire from the command line',flush=True)

    ### parse the main command line arguments 
    parser = argparse.ArgumentParser()

    # commonly used known args for sapphire
    # NOTE: add more here as required
    parser.add_argument("--path_config", type=str, default=None, help='path to default baseline yaml config file', required=True)
    parser.add_argument("--mock_num", type=int)
    parser.add_argument("--rng_sample", type=int)
    parser.add_argument("--rng_init", type=int)

    # parse the known and unknown args
    # NOTE: add function to parse unknown args, assuming they follow same --key value structure on command line
    args, unknown_args = parser.parse_known_args()

    # convert args Namespace to dict 
    config = vars(args)

    # finally call sapphire.run as usual
    run(config)


if __name__ == '__main__':
    
    main()
