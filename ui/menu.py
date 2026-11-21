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

from q_learning import ACTIONS


class MenuMixin:
    def create_main_menu(self):
        # Clear existing widgets if coming back from a sub-menu
        for widget in self.root.winfo_children():
            if widget != self.hp_frame: # Don't destroy hyperparameter frame
                widget.destroy()

        self.main_menu_frame = ttk.Frame(self.root, padding="20")
        self.main_menu_frame.pack(expand=True, fill="both")

        ttk.Label(self.main_menu_frame, text="Choose a Pathfinding Mode:", font=("Arial", 16)).pack(pady=20)

        ttk.Button(self.main_menu_frame, text="Original Grid Pathfinding",
                   command=self.show_original_grid_interface, width=30, style='TButton').pack(pady=10)
        ttk.Button(self.main_menu_frame, text="Real-World Map Pathfinding",
                   command=self.show_real_world_map_interface, width=30, style='TButton').pack(pady=10)
        ttk.Button(self.main_menu_frame, text="Utilities",
                   command=self.show_utilities_interface, width=30, style='TButton').pack(pady=10)


    def setup_hyperparameters_frame(self):
        # Hyperparameters (always visible at the bottom)
        self.hp_frame = ttk.LabelFrame(self.root, text="Hyperparameters", padding="10")
        self.hp_frame.pack(side="bottom", fill="x", pady=5, padx=10)

        # Learning Rate
        ttk.Label(self.hp_frame, text="Learning Rate:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.lr_var = tk.DoubleVar(value=0.1)
        ttk.Entry(self.hp_frame, textvariable=self.lr_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        # Discount Factor
        ttk.Label(self.hp_frame, text="Discount Factor:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        self.df_var = tk.DoubleVar(value=0.99)
        ttk.Entry(self.hp_frame, textvariable=self.df_var, width=10).grid(row=0, column=3, padx=5, pady=2)

        # Epsilon Decay Rate
        ttk.Label(self.hp_frame, text="Epsilon Decay Rate:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.edr_var = tk.DoubleVar(value=0.9995)
        ttk.Entry(self.hp_frame, textvariable=self.edr_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        # Min Epsilon
        ttk.Label(self.hp_frame, text="Min Epsilon:").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        self.min_eps_var = tk.DoubleVar(value=0.01)
        ttk.Entry(self.hp_frame, textvariable=self.min_eps_var, width=10).grid(row=1, column=3, padx=5, pady=2)

        # Total Episodes
        ttk.Label(self.hp_frame, text="Total Episodes:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.episodes_var = tk.IntVar(value=5000)
        ttk.Entry(self.hp_frame, textvariable=self.episodes_var, width=10).grid(row=2, column=1, padx=5, pady=2)

        # Configure columns to expand nicely
        for i in range(4):
            self.hp_frame.grid_columnconfigure(i, weight=1)


    def show_original_grid_interface(self):
        self.main_menu_frame.pack_forget() # Hide main menu

        self.original_grid_frame = ttk.Frame(self.root, padding="10")
        self.original_grid_frame.pack(expand=True, fill="both")

        # Top control frame
        control_frame = ttk.Frame(self.original_grid_frame)
        control_frame.pack(pady=5, fill="x")
        ttk.Button(control_frame, text="< Back to Main Menu", command=self.back_to_main_menu).pack(side="left", padx=5)

        # Configuration frame for grid
        config_frame = ttk.LabelFrame(control_frame, text="Grid Config", padding="10")
        config_frame.pack(side="left", padx=10, fill="x", expand=True)

        ttk.Label(config_frame, text="Size (4-8):").grid(row=0, column=0, padx=2, pady=1, sticky="w")
        self.grid_size_var = tk.IntVar(value=8)
        ttk.Spinbox(config_frame, from_=4, to=8, textvariable=self.grid_size_var, width=4).grid(row=0, column=1, padx=2, pady=1, sticky="ew")

        ttk.Label(config_frame, text="Obstacles:").grid(row=1, column=0, padx=2, pady=1, sticky="w")
        self.num_obstacles_var = tk.IntVar(value=8)
        ttk.Spinbox(config_frame, from_=0, to=31, textvariable=self.num_obstacles_var, width=4).grid(row=1, column=1, padx=2, pady=1, sticky="ew")

        ttk.Button(config_frame, text="Generate Grid & Train", command=self.run_original_grid_training).grid(row=0, column=2, rowspan=2, padx=5, pady=2)
        config_frame.grid_columnconfigure(1, weight=1)

        # Display buttons for Q-Table and Graph
        display_buttons_frame = ttk.Frame(control_frame)
        display_buttons_frame.pack(side="right", padx=5)
        ttk.Button(display_buttons_frame, text="Show Grid Map", command=self.show_original_grid_map).pack(side="top", pady=2)
        ttk.Button(display_buttons_frame, text="Show Q-Table", command=lambda: self.show_q_table_popup("original")).pack(side="top", pady=2)


        # Progress Bar
        self.progress_label_og = ttk.Label(self.original_grid_frame, text="Training Progress: 0%")
        self.progress_label_og.pack(pady=2)
        self.progress_og = ttk.Progressbar(self.original_grid_frame, orient="horizontal", length=400, mode="determinate")
        self.progress_og.pack(pady=5, fill="x", padx=5)

        # Canvas for grid visualization
        self.grid_canvas = tk.Canvas(self.original_grid_frame, bg="white", borderwidth=2, relief="groove")
        self.grid_canvas.pack(expand=True, fill="both", pady=10)
        self.grid_canvas_size = 500 # Max size for grid visualization
        self.grid_canvas.bind("<Configure>", self.on_grid_canvas_resize)

        # Matplotlib Figure for Plotting (Learning Curve)
        self.fig_og, self.ax_og = plt.subplots(figsize=(6, 2.5), dpi=100) # Smaller figure for learning curve
        self.canvas_og_mpl = FigureCanvasTkAgg(self.fig_og, master=self.original_grid_frame)
        self.canvas_og_mpl_widget = self.canvas_og_mpl.get_tk_widget()
        self.canvas_og_mpl_widget.pack(side="bottom", fill="x", pady=5)
        self.plot_learning_curve([], self.ax_og, self.canvas_og_mpl, "Original Grid") # Initial blank plot

        self.last_drawn_grid_params = None # To track if grid needs redrawing


    def on_grid_canvas_resize(self, event):
        # Redraw the grid if the canvas size changes and we have grid parameters
        if self.last_drawn_grid_params:
            self.draw_grid_on_canvas(self.last_drawn_grid_params[0],
                                     self.last_drawn_grid_params[1],
                                     self.last_drawn_grid_params[2],
                                     self.last_drawn_grid_params[3])


    def show_original_grid_map(self):
        """Shows the current grid configuration on the canvas."""
        if self.current_grid_params:
            grid_size, start_state, end_state, obstacles = self.current_grid_params
            self.draw_grid_on_canvas(grid_size, start_state, end_state, obstacles)
            messagebox.showinfo("Original Grid Map", "Current grid configuration displayed on canvas.")
        else:
            messagebox.showwarning("No Grid", "Please generate a grid first by clicking 'Generate Grid & Train'.")


    def show_real_world_map_interface(self):
        self.main_menu_frame.pack_forget() # Hide main menu

        self.real_world_frame = ttk.Frame(self.root, padding="10")
        self.real_world_frame.pack(expand=True, fill="both")

        # Top control frame
        control_frame_rw = ttk.Frame(self.real_world_frame)
        control_frame_rw.pack(pady=5, fill="x")
        ttk.Button(control_frame_rw, text="< Back to Main Menu", command=self.back_to_main_menu).pack(side="left", padx=5)

        # Place name input
        place_frame = ttk.Frame(control_frame_rw)
        place_frame.pack(side="left", padx=10, fill="x", expand=True)

        ttk.Label(place_frame, text="Place Name (e.g., 'Berkeley, CA' or 'Random Location'):").grid(row=0, column=0, sticky="w", padx=5)
        self.place_name_var = tk.StringVar(value="San Francisco, CA") # Changed default for better testing
        self.place_name_entry = ttk.Entry(place_frame, textvariable=self.place_name_var, width=40)
        self.place_name_entry.grid(row=0, column=1, padx=5, sticky="ew")
        ttk.Button(place_frame, text="Load Map & Train", command=self.run_real_world_training).grid(row=0, column=2, padx=5)
        place_frame.grid_columnconfigure(1, weight=1)

        # Display buttons for Q-Table and Graph
        display_buttons_frame_rw = ttk.Frame(control_frame_rw)
        display_buttons_frame_rw.pack(side="right", padx=5)
        ttk.Button(display_buttons_frame_rw, text="Show OSMnx Graph", command=self.show_osmnx_graph).pack(side="top", pady=2)
        ttk.Button(display_buttons_frame_rw, text="Show Q-Table", command=lambda: self.show_q_table_popup("real_world")).pack(side="top", pady=2)

        # Progress Bar
        self.progress_label_rw = ttk.Label(self.real_world_frame, text="Training Progress: 0%")
        self.progress_label_rw.pack(pady=2)
        self.progress_rw = ttk.Progressbar(self.real_world_frame, orient="horizontal", length=400, mode="determinate")
        self.progress_rw.pack(pady=5, fill="x", padx=5)

        # Matplotlib Figure for Map and Learning Curve
        # We'll use one figure, potentially with subplots, or just the map and a smaller curve.
        self.fig_rw, (self.ax_rw_map, self.ax_rw_curve) = plt.subplots(2, 1, figsize=(30, 25), gridspec_kw={'height_ratios': [3, 1]}, dpi=100)
        self.canvas_rw_mpl = FigureCanvasTkAgg(self.fig_rw, master=self.real_world_frame)
        self.canvas_rw_mpl_widget = self.canvas_rw_mpl.get_tk_widget()
        self.canvas_rw_mpl_widget.pack(expand=True, fill="both", pady=10)

        # Initial blank map and curve
        self.ax_rw_map.set_title("Real-World Map Visualization")
        self.ax_rw_map.set_xticks([])
        self.ax_rw_map.set_yticks([])
        self.plot_learning_curve([], self.ax_rw_curve, self.canvas_rw_mpl, "Real-World Grid") # Initial blank plot


    def show_osmnx_graph(self):
        if not self.current_osm_graph:
            messagebox.showwarning("No Map Loaded", "Please load a real-world map first by clicking 'Load Map & Train'.")
            return

        # Plot the graph in a new window using osmnx's plotting capabilities
        fig, ax = ox.plot_graph(self.current_osm_graph, show=False, close=False,
                                bgcolor='lightgray', edge_color='white', node_size=10,
                                node_color='blue', edge_linewidth=0.5,
                                figsize=(20, 20)) # Fixed size for pop-up

        # Highlight start and end nodes if available
        if self.real_world_graph_start_node and self.real_world_graph_end_node:
            s_node = self.real_world_graph_start_node
            e_node = self.real_world_graph_end_node

            if s_node in self.current_osm_graph.nodes:
                sx, sy = self.current_osm_graph.nodes[s_node]['x'], self.current_osm_graph.nodes[s_node]['y']
                ax.scatter(sx, sy, color='green', s=200, zorder=5, edgecolors='black', label='Start Node')
                ax.text(sx, sy, 'S', fontsize=12, ha='center', va='center', color='white', zorder=6)

            if e_node in self.current_osm_graph.nodes:
                ex, ey = self.current_osm_graph.nodes[e_node]['x'], self.current_osm_graph.nodes[e_node]['y']
                ax.scatter(ex, ey, color='red', s=200, zorder=5, edgecolors='black', label='End Node')
                ax.text(ex, ey, 'E', fontsize=12, ha='center', va='center', color='white', zorder=6)

        ax.set_title(f"OSMnx Graph for {self.place_name_var.get()}")
        plt.show() # Display the matplotlib figure in a new window


    def show_utilities_interface(self):
        self.main_menu_frame.pack_forget() # Hide main menu

        self.utilities_frame = ttk.Frame(self.root, padding="10")
        self.utilities_frame.pack(expand=True, fill="both")

        # Back button
        ttk.Button(self.utilities_frame, text="< Back to Main Menu", command=self.back_to_main_menu).pack(pady=5, anchor="nw")

        # Frame for Utilities
        utilities_content_frame = ttk.Frame(self.utilities_frame, padding="10")
        utilities_content_frame.pack(pady=10, padx=10, expand=True, fill="both")

        # Working Directory Button
        ttk.Button(utilities_content_frame, text="Print Working Directory", command=self.print_working_dir).pack(pady=5)
        self.working_dir_label = ttk.Label(utilities_content_frame, text="")
        self.working_dir_label.pack(pady=2)

        # File Upload Simulation Button
        ttk.Button(utilities_content_frame, text="Simulate File Upload", command=self.handle_file_upload).pack(pady=5)

        # Text entry for string manipulation
        ttk.Label(utilities_content_frame, text="String 1:").pack(pady=2)
        self.string1_entry = ttk.Entry(utilities_content_frame, width=50)
        self.string1_entry.insert(tk.END, "This is the first string.")
        self.string1_entry.pack(pady=2)

        ttk.Label(utilities_content_frame, text="String 2 (chars to remove):").pack(pady=2)
        self.string2_entry = ttk.Entry(utilities_content_frame, width=50)
        self.string2_entry.insert(tk.END, "aeiou")
        self.string2_entry.pack(pady=2)

        ttk.Button(utilities_content_frame, text="Remove Chars", command=self.remove_chars_action).pack(pady=5)

        ttk.Label(utilities_content_frame, text="Result:").pack(pady=2)
        self.remove_chars_result_label = ttk.Label(utilities_content_frame, text="")
        self.remove_chars_result_label.pack(pady=2)


    def show_q_table_popup(self, mode):
        if self.current_q_table is None:
            messagebox.showwarning("No Q-Table", "No Q-Table available. Please run a training simulation first.")
            return

        popup = tk.Toplevel(self.root)
        popup.title(f"{mode.replace('_', ' ').title()} Q-Table")
        popup.geometry("600x400")

        # Create a scrollable text widget
        text_frame = ttk.Frame(popup)
        text_frame.pack(expand=True, fill="both", padx=10, pady=10)

        text_widget = tk.Text(text_frame, wrap="none", font=("Consolas", 10))
        text_widget.pack(side="left", expand=True, fill="both")

        vsb = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        vsb.pack(side="right", fill="y")
        text_widget.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(popup, orient="horizontal", command=text_widget.xview)
        hsb.pack(side="bottom", fill="x")
        text_widget.configure(xscrollcommand=hsb.set)

        # Format Q-table as a string
        q_table_str = "Q-Table:\n"
        grid_size = self.current_q_table.shape[0]
        for r in range(grid_size):
            for c in range(grid_size):
                q_table_str += f"\nState ({r}, {c}):\n"
                for action_idx, action_q_value in enumerate(self.current_q_table[r, c]):
                    action_desc = f"Action {action_idx} {ACTIONS[action_idx]}"
                    q_table_str += f"  {action_desc}: {action_q_value:.4f}\n"
            q_table_str += "-" * 50 + "\n" # Separator between rows

        text_widget.insert(tk.END, q_table_str)
        text_widget.config(state="disabled") # Make text read-only


    def back_to_main_menu(self):
        if hasattr(self, 'original_grid_frame') and self.original_grid_frame.winfo_exists():
            self.original_grid_frame.destroy()
        if hasattr(self, 'real_world_frame') and self.real_world_frame.winfo_exists():
            self.real_world_frame.destroy()
        if hasattr(self, 'utilities_frame') and self.utilities_frame.winfo_exists():
            self.utilities_frame.destroy()

        # Stop any ongoing animation threads
        self.animation_stop_event.set()
        if self.animation_thread and self.animation_thread.is_alive():
            print("Joining animation thread...")
            self.animation_thread.join(timeout=0.5) # Give it a moment to stop gracefully
            print("Animation thread joined.")
        self.animation_stop_event.clear() # Reset for next use

        self.current_q_table = None
        self.current_grid_params = None
        self.current_tab_trained = None
        self.current_osm_graph = None
        self.real_world_grid_to_node_map = None
        self.real_world_node_to_grid_map = None
        self.real_world_min_coords = None
        self.real_world_max_coords = None
        self.real_world_graph_start_node = None
        self.real_world_graph_end_node = None

        # Close all matplotlib figures
        plt.close('all')

        self.create_main_menu() # Re-create the main menu