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

from utilities import get_working_directory, simulate_file_upload, remove_dirty_chars


class UtilitiesMixin:
    def print_working_dir(self):
        directory = get_working_directory()
        self.working_dir_label.config(text=f"Current Working Directory: {directory}")
        messagebox.showinfo("Working Directory", f"Current Working Directory: {directory}")


    def handle_file_upload(self):
        uploaded_files = simulate_file_upload()
        if uploaded_files:
            file_name = list(uploaded_files.keys())[0]
            content = uploaded_files[file_name].decode('utf-8')
            messagebox.showinfo("Simulated Upload Result", f"File '{file_name}' uploaded. Content preview:\n{content[:100]}...")
        else:
            messagebox.showinfo("Simulated Upload Result", "No file simulated for upload.")


    def remove_chars_action(self):
        s1 = self.string1_entry.get()
        s2 = self.string2_entry.get()
        result = remove_dirty_chars(s1, s2)
        self.remove_chars_result_label.config(text=f"Removed: '{result}'")
        messagebox.showinfo("Remove Characters", f"Original: '{s1}'\nChars to remove: '{s2}'\nResult: '{result}'")