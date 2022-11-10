
import os
from pathlib import Path
import toml
import json
from typing import Dict, Any
import logging

# define paths to files and outputs
# define helpful paths 
PATH_TO_THIS_FILE: Path = Path(__file__).resolve()
ABSPATH_TO_TOML: Path = PATH_TO_THIS_FILE.parent / "config.toml"
PATH_TO_MP4S: Path = PATH_TO_THIS_FILE.parent / "single-agent" / "mp4s"

# use toml.load to read `config.toml` file in as a dictionary
CONFIG_DICT: Dict[Any, Any] = toml.load(str(ABSPATH_TO_TOML))

class MDP:
    episodes: int = CONFIG_DICT['MDP']['general']['episodes']
    wildfire_update_window: int = CONFIG_DICT['MDP']['general']['wildfire_update_window']
    aircraft_update_window: int = CONFIG_DICT['MDP']['general']['aircraft_update_window']
    hotshot_update_window: int = CONFIG_DICT['MDP']['general']['hotshot_update_window']
    stochastic_fire_spread: bool = CONFIG_DICT['MDP']['general']['stochastic_fire_spread']
    generate_plots: bool = CONFIG_DICT['MDP']['plotting']['generate_plots']
    generate_mp4: bool = CONFIG_DICT['MDP']['plotting']['generate_mp4']
    board_increase: int = CONFIG_DICT['MDP']['plotting']['board_increase']
    colormap: str = CONFIG_DICT['MDP']['plotting']['colormap']

class MODEL_FREE:
    generate_plots: bool = CONFIG_DICT['MODEL_FREE']['plotting']['generate_plots']
    generate_mp4: bool = CONFIG_DICT['MODEL_FREE']['plotting']['generate_mp4']

# TODO add logger 
