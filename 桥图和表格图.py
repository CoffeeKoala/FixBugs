# from pytalos.client import AsyncTalosClient
import numpy as np
import pandas as pd
import os,time,sys
import seaborn as sns
import scipy.stats.contingency as cy 
from sklearn.cluster import KMeans
import math
import operator
import scipy.stats
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdate

from datetime import datetime
import networkx as nx
import Levenshtein

import ChromaPalette as CP
from IPython.display import display_html

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

color_list = CP.chroma_palette.color_palette(name='Sunrise',N=5)
