"""
-------------------------------
| Dartmouth College           |
| ENGG 199.09 - Game Theory   |
| Fall 2022                   |
-------------------------------

Routines for envs.py and models.py

This script will be used to store utility functions

This script can be run from the command line using: $ routines.py
"""

from typing import Iterator, Optional, Tuple
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# local imports 
from settings import Configs


class Solution:

    # NOTE: the idea for using a Solution object came from CS276 @ Dartmouth
    # A few lines of this code were copied form an old homework from that class. (S. Bertsch Oct. 20, 2022)
    def __init__(self, problem_name: str, model_name: str):
        self.problem_name = problem_name
        self.model_name = model_name
        self.path = []
        self.nodes_visited = 0
        self.cost = 0
        self.solved = False
        self.reward = 0
        self.steps = 0

    def __repr__(self):
        s_large = '-'*50
        s_small = '-'*10
        s = f'{s_large} \n {s_small} Problem: {self.problem_name} {s_small} \n'
        s += f'{s_small} Model: {self.model_name} {s_small} \n'

        if len(self.path) > 0: 
            s += f'Nodes visited: {self.nodes_visited} \n'
            s += f'Total steps: {self.steps} \n'
            s += f'Final reward: {self.reward} \n'
            s += f'Solved: {self.solved} \n{s_large}'

        else:
            s += f'{self.model_name} failed to find a solution after searching {self.nodes_visited} nodes. \n{s_large}'

        return s


def write_animation(
    itr: Iterator[np.array],
    out_file: Path,
    dpi: int = 50,
    fps: int = 30,
    title: str = "Animation",
    comment: Optional[str] = None,
    writer: str = "ffmpeg") -> None:
    """Function that writes an animation from a stream of input tensors.

    Source: https://ben.bolte.cc/matplotlib-videos

    Args:
        itr: The image iterator, yielding images with shape (H, W, C).
        out_file: The path to the output file.
        dpi: Dots per inch for output image.
        fps: Frames per second for the video.
        title: Title for the video metadata.
        comment: Comment for the video metadata.
        writer: The Matplotlib animation writer to use (if you use the
            default one, make sure you have `ffmpeg` installed on your
            system).
    """

    first_img = next(itr)
    height, width = first_img.shape
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi))
    plt.set_cmap(Configs.COLORMAP)

    # Ensures that there's no extra space around the image.
    fig.subplots_adjust(
        left=0,
        bottom=0,
        right=1,
        top=1,
        wspace=None,
        hspace=None,
    )

    # Creates the writer with the given metadata.
    Writer = animation.writers[writer]
    metadata = {
        "title": title,
        "artist": __name__,
        "comment": comment,
    }
    mpl_writer = Writer(
        fps=fps,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )

    with mpl_writer.saving(fig, out_file, dpi=dpi):
        im = ax.imshow(first_img, interpolation="nearest")
        mpl_writer.grab_frame()

        for img in itr:
            im.set_data(img)
            mpl_writer.grab_frame()

    print(f'Movie file written to the following locaiton: \n {str(out_file)}')

if __name__ == "__main__":
    pass
