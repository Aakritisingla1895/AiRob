import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time  # to time the learning process
import json  # to get the configuration of the environment
from environments.simple_road_env import Road
from brains.simple_brains import MC
from brains.simple_brains import QLearningTable
from brains.simple_brains import SarsaTable
from brains.simple_brains import ExpectedSarsa
from brains.simple_brains import SarsaLambdaTable
from brains.simple_brains import DP
from brains.simple_DQN_tensorflow import DeepQNetwork
from brains.simple_DQN_pytorch import Agent
from collections import deque
import math
from utils.logger import Logger