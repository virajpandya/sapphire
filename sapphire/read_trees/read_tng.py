"""
This module reads in one or multiple subvolume tree files for the TNG50/100/300 
simulation using ytree. The output files are in the original consistent-trees ASCII
format (Rodriguez-Puebla+16).

This script should also filter out halos below requested Mvir resolution limit, bad snapshots, etc.

This script should also compute any additional quantities needed by sapphire 
such as Vvir=sqrt(GMvir/Rvir) if not already available in the tree file.
"""


