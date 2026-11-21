import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import osmnx as ox
import threading
import time
import subprocess
import collections
import sys

from ui.menu import MenuMixin
from ui.utilities import UtilitiesMixin
from ui.grid import GridMixin
from ui.real_world import RealWorldMixin


class App(MenuMixin, UtilitiesMixin, GridMixin, RealWorldMixin):
    def __init__(self, root):
        self.root = root
        root.title("Q-Learning Pathfinding Simulator")
        root.geometry("800x700") # Initial size

        self.current_q_table = None
        self.current_grid_params = None # (grid_size, start_state, end_state, obstacles)
        self.current_tab_trained = None # "original" or "real_world"

        # For Real-World Map specific data
        self.current_osm_graph = None
        self.real_world_grid_to_node_map = None
        self.real_world_node_to_grid_map = None
        self.real_world_min_coords = None
        self.real_world_max_coords = None
        self.real_world_graph_start_node = None # Store original graph start node
        self.real_world_graph_end_node = None   # Store original graph end node

        self.animation_thread = None
        self.animation_stop_event = threading.Event()

        self.create_main_menu()
        self.setup_hyperparameters_frame()