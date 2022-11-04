"""
-------------------------------
| Dartmouth College           |
| ENGG 199.09 - Game Theory   |
| Fall 2022                   |
-------------------------------

This file will store local configs, paths, and settings  

"""

# imports 
from pathlib import Path

# define helpful paths 
PATH_TO_THIS_FILE: Path = Path(__file__).resolve()
PATH_TO_MP4S: Path = PATH_TO_THIS_FILE.parent / "mp4s"

class Configs:

    # used to determine size of the video output
    BOARD_INCREASE = 150

    # matplotlib.org/stable/tutorials/colors/colormaps.html
    COLORMAP = 'cool'  # choose from the following: ['cool', 'rainbow', 'turbo', 'viridis', 'plasma']

    generate_video: bool = True

