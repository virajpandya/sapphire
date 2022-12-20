"""
This file is run automatically if the user does "python -m sapphire <parameters.JSON> on the command line
NOTE: *must* include the -m flag in the python call so it runs as a python package and explicit relative imports work.
"""

from .driver import run # might be redundant with __init__.py 

# argparse the input.JSON file, convert into a "parameters" dict, and call run(parameters) automatically
# run(parameters)

print("You've reached __main__.py! Command line runs are not yet implemented.")