"""
This module reads the raw consistent-trees merger trees of the core FIRE-2 halos
created by Viraj for Pandya+20. These files are each huge (~couple GB) so should
probably be analyzed on a cluster.

This script should also compute any additional quantities needed by sapphire 
such as Vvir=sqrt(GMvir/Rvir) if not already available in the tree file.
"""

