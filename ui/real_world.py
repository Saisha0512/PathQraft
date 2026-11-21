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

from q_learning import train_q_learning, find_optimal_path
from map_utils import get_real_world_graph, get_start_end_nodes, graph_to_simplified_grid


class RealWorldMixin:
    def run_real_world_training(self):
        # Stop any ongoing animation
        self.animation_stop_event.set()
        if self.animation_thread and self.animation_thread.is_alive():
            self.animation_thread.join(timeout=0.2) # Short timeout
        self.animation_stop_event.clear() # Reset for next use

        self.clear_real_world_map_plot() # Clear previous map and plot
        self.progress_rw['value'] = 0
        self.progress_label_rw['text'] = "Training Progress: 0%"

        place_name = self.place_name_var.get().strip()
        if not place_name:
            messagebox.showwarning("Input Error", "Please enter a place name or 'Random Location'.")
            return

        # Get hyperparameters from GUI
        try:
            learning_rate = self.lr_var.get()
            discount_factor = self.df_var.get()
            epsilon_decay_rate = self.edr_var.get()
            min_epsilon = self.min_eps_var.get()
            total_episodes = self.episodes_var.get()
            if not (0 < learning_rate <= 1 and 0 <= discount_factor <= 1 and 0 < epsilon_decay_rate <= 1 and 0 <= min_epsilon < 1 and total_episodes > 0):
                raise ValueError("Hyperparameters out of valid range.")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Please enter valid numeric values for hyperparameters. Error: {e}")
            return

        # Run graph loading and training in a separate thread
        def real_world_thread():
            self.root.after(0, lambda: messagebox.showinfo("Loading Map & Training", f"Starting to load map for '{place_name}' and train Q-Learning. This may take some time depending on map size and internet speed."))
            self.root.after(0, lambda: self.ax_rw_map.text(0.5, 0.5, "Loading Map...", ha='center', va='center', transform=self.ax_rw_map.transAxes))
            self.root.after(0, self.canvas_rw_mpl.draw)

            graph = get_real_world_graph(place_name=place_name)
            if not graph:
                self.root.after(0, lambda: self.update_progress(0, 1, "real_world"))
                self.root.after(0, self.clear_real_world_map_plot)
                return

            start_node_graph, end_node_graph = get_start_end_nodes(graph)
            if not start_node_graph or not end_node_graph:
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not select valid start/end nodes from the map. Map might be too small or disconnected."))
                self.root.after(0, lambda: self.update_progress(0, 1, "real_world"))
                self.root.after(0, self.clear_real_world_map_plot)
                return

            # Store the original graph
            self.current_osm_graph = graph
            self.real_world_graph_start_node = start_node_graph
            self.real_world_graph_end_node = end_node_graph

            # Convert graph to a simplified grid
            REAL_WORLD_GRID_SIZE = 30 # Can be adjusted, user requested "low grid size"
            grid_size_real, start_state_real, end_state_real, obstacles_real, \
            grid_to_node_map, node_to_grid_map, min_coords, max_coords, _, _ = \
                graph_to_simplified_grid(graph, start_node_graph, end_node_graph, grid_size=REAL_WORLD_GRID_SIZE)

            if grid_size_real is None:
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not create a valid grid from the map. Check console for details."))
                self.root.after(0, lambda: self.update_progress(0, 1, "real_world"))
                self.root.after(0, self.clear_real_world_map_plot)
                return

            self.current_grid_params = (grid_size_real, start_state_real, end_state_real, obstacles_real)
            self.real_world_grid_to_node_map = grid_to_node_map
            self.real_world_node_to_grid_map = node_to_grid_map
            self.real_world_min_coords = min_coords
            self.real_world_max_coords = max_coords

            # Train Q-Learning on the real-world simplified grid
            q_table_real, episode_rewards_real = train_q_learning(
                grid_size=grid_size_real,
                start_state=start_state_real,
                end_state=end_state_real,
                obstacles=obstacles_real,
                total_episodes=total_episodes,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon_decay_rate=epsilon_decay_rate,
                min_epsilon=min_epsilon,
                progress_callback=lambda current, total: self.update_progress(current, total, "real_world")
            )

            if q_table_real is not None:
                self.current_q_table = q_table_real
                self.current_tab_trained = "real_world"
                self.root.after(0, lambda: self.plot_learning_curve(episode_rewards_real, self.ax_rw_curve, self.canvas_rw_mpl, f"Real-World Grid ({place_name})"))
                self.root.after(0, lambda: messagebox.showinfo("Training Complete", f"Q-Learning training finished for Real-World Grid ('{place_name}'). Visualizing path..."))
                self.root.after(0, self.animate_real_world_path) # Start animation
            else:
                 self.root.after(0, lambda: messagebox.showerror("Training Failed", f"Q-Learning training for Real-World Grid ('{place_name}') failed or no path found. Check console for details."))
                 self.root.after(0, lambda: self.plot_learning_curve([], self.ax_rw_curve, self.canvas_rw_mpl, f"Real-World Grid ({place_name})"))
                 self.root.after(0, self.clear_real_world_map_plot)


        self.animation_thread = threading.Thread(target=real_world_thread)
        self.animation_thread.daemon = True
        self.animation_thread.start()


    def clear_real_world_map_plot(self):
        self.ax_rw_map.clear()
        self.ax_rw_map.set_title("Real-World Map Visualization")
        self.ax_rw_map.set_xticks([])
        self.ax_rw_map.set_yticks([])
        self.ax_rw_curve.clear()
        self.ax_rw_curve.set_title("Real-World Grid Learning Curve")
        self.ax_rw_curve.set_xlabel('Episode')
        self.ax_rw_curve.set_ylabel('Total Reward')
        self.ax_rw_curve.grid(True)
        self.canvas_rw_mpl.draw()


    def draw_real_world_map(self, current_agent_pos_grid=None, optimal_path_grid=None):
        if not self.current_osm_graph or not self.current_grid_params:
            return

        grid_size, start_state_grid, end_state_grid, obstacles_grid = self.current_grid_params
        graph = self.current_osm_graph
        grid_to_node_map = self.real_world_grid_to_node_map
        min_x, min_y = self.real_world_min_coords
        max_x, max_y = self.real_world_max_coords

        self.ax_rw_map.clear()
        self.ax_rw_map.set_title("Real-World Map Pathfinding")
        self.ax_rw_map.set_xticks([])
        self.ax_rw_map.set_yticks([])

        # Plot the base map with minimal objects (edges and relevant nodes)
        ox.plot_graph(graph, ax=self.ax_rw_map, show=False, close=False,
                      bgcolor='white', edge_color='lightgray', node_size=0, edge_linewidth=0.5)

        # Plot the obstacles as shaded regions
        cell_width_map = (max_x - min_x) / grid_size
        cell_height_map = (max_y - min_y) / grid_size

        def grid_to_map_center_coords(r, c):
            # Map grid cell (r, c) to center of its corresponding real-world coordinate box
            map_x = min_x + c * cell_width_map + cell_width_map / 2
            map_y = max_y - r * cell_height_map - cell_height_map / 2 # Invert row for map Y (higher row = lower lat)
            return map_x, map_y

        # Plot obstacles on the map (as semi-transparent rectangles)
        for r, c in obstacles_grid:
            map_x_rect_origin = min_x + c * cell_width_map
            map_y_rect_origin = max_y - (r + 1) * cell_height_map # y-origin for rectangle is bottom-left corner
            self.ax_rw_map.add_patch(plt.Rectangle((map_x_rect_origin, map_y_rect_origin),
                                                 cell_width_map, cell_height_map, color='darkgrey', alpha=0.5))

        # Plot Start and End Nodes (larger, distinct colors)
        # Using the original graph nodes for more accurate positioning
        if self.real_world_graph_start_node and self.real_world_graph_start_node in graph.nodes:
            s_node = self.real_world_graph_start_node
            s_x, s_y = graph.nodes[s_node]['x'], graph.nodes[s_node]['y']
            self.ax_rw_map.scatter(s_x, s_y, color='green', s=150, zorder=5, label='Start', edgecolors='black')
            self.ax_rw_map.text(s_x, s_y, 'S', fontsize=10, ha='center', va='center', color='white', zorder=6)

        if self.real_world_graph_end_node and self.real_world_graph_end_node in graph.nodes:
            e_node = self.real_world_graph_end_node
            e_x, e_y = graph.nodes[e_node]['x'], graph.nodes[e_node]['y']
            self.ax_rw_map.scatter(e_x, e_y, color='red', s=150, zorder=5, label='End', edgecolors='black')
            self.ax_rw_map.text(e_x, e_y, 'E', fontsize=10, ha='center', va='center', color='white', zorder=6)


        # Plot the optimal path if provided
        if optimal_path_grid:
            # Convert grid path to a sequence of real-world coordinates for plotting
            real_path_coords = []
            for grid_cell in optimal_path_grid:
                # Get the map coordinates for the center of this grid cell
                map_x, map_y = grid_to_map_center_coords(grid_cell[0], grid_cell[1])
                real_path_coords.append((map_x, map_y))

            if len(real_path_coords) > 1:
                # Draw lines connecting these map coordinates
                xs = [p[0] for p in real_path_coords]
                ys = [p[1] for p in real_path_coords]
                self.ax_rw_map.plot(xs, ys, color='blue', linewidth=3, solid_capstyle='round', zorder=4, label='Optimal Path')
            elif len(real_path_coords) == 1: # Just a point if path is only start/end or very short
                 x, y = real_path_coords[0]
                 self.ax_rw_map.scatter(x, y, color='blue', s=50, zorder=4, alpha=0.7)


        # Plot current agent position
        if current_agent_pos_grid:
            agent_x, agent_y = grid_to_map_center_coords(current_agent_pos_grid[0], current_agent_pos_grid[1])
            self.ax_rw_map.scatter(agent_x, agent_y, color='gold', s=200, zorder=10, marker='o', edgecolors='black') # Agent marker

        self.fig_rw.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust layout to make space for title/footer
        self.canvas_rw_mpl.draw()


    def animate_real_world_path(self):
        if self.current_q_table is None or self.current_grid_params is None or self.current_tab_trained != "real_world":
            messagebox.showwarning("No Data", "Please load a map and train the Q-Learning model for the Real-World Map first.")
            return

        grid_size, start_state_grid, end_state_grid, obstacles_grid = self.current_grid_params
        optimal_path_grid = find_optimal_path(self.current_q_table, grid_size, start_state_grid, end_state_grid, obstacles_grid)

        if not optimal_path_grid:
            messagebox.showwarning("Path Not Found", "Optimal path could not be determined for the real-world map. Agent might be stuck or no path exists.")
            return

        # Ensure base map is drawn before animation
        self.root.after(0, lambda: self.draw_real_world_map(current_agent_pos_grid=None, optimal_path_grid=None))

        # Animation loop
        def animate():
            for i, state_grid in enumerate(optimal_path_grid):
                if self.animation_stop_event.is_set():
                    print("Animation stopped by user.")
                    break
                # Update map with current agent position
                self.root.after(0, lambda s_grid=state_grid: self.draw_real_world_map(current_agent_pos_grid=s_grid, optimal_path_grid=None))
                time.sleep(0.3) # Control animation speed

            if not self.animation_stop_event.is_set():
                # After animation, draw the final path clearly and remove agent marker
                self.root.after(0, lambda: self.draw_real_world_map(current_agent_pos_grid=None, optimal_path_grid=optimal_path_grid))
                messagebox.showinfo("Animation Complete", "Agent has finished traversing the optimal path on the real-world map.")

        self.animation_thread = threading.Thread(target=animate)
        self.animation_thread.daemon = True
        self.animation_thread.start()