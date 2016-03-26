""" Check whih functions take the most time. """
import os
import pstats

os.system('python -m cProfile -o restats localricci.py')

with open('profile.txt', 'w') as f:
    p = pstats.Stats('restats', stream=f)
    p.strip_dirs().sort_stats("cumtime").print_stats(100)

os.unlink('restats')
