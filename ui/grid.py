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

from q_learning import train_q_learning, find_optimal_path, has_path


class GridMixin:
    def update_progress(self, current_episode, total_episodes, tab_name="original"):
        progress_percentage = (current_episode / total_episodes) * 100
        if tab_name == "original":
            self.progress_og['value'] = progress_percentage
            self.progress_label_og['text'] = f"Training Progress: {progress_percentage:.1f}%"
        elif tab_name == "real_world":
            self.progress_rw['value'] = progress_percentage
            self.progress_label_rw['text'] = f"Training Progress: {progress_percentage:.1f}%"
        self.root.update_idletasks() # Update the GUI


    def plot_learning_curve(self, episode_rewards, ax, canvas, title):
        ax.clear()
        if episode_rewards:
            ax.plot(episode_rewards)
            ax.set_title(f'{title} Learning Curve')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
        else:
            ax.text(0.5, 0.5, 'No episode rewards to plot', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'{title} Learning Curve')
        ax.grid(True)
        canvas.draw()


    def run_original_grid_training(self):
        # Stop any ongoing animation
        self.animation_stop_event.set()
        if self.animation_thread and self.animation_thread.is_alive():
            self.animation_thread.join(timeout=0.2) # Short timeout
        self.animation_stop_event.clear() # Reset for next use

        self.clear_original_grid_canvas() # Clear previous grid path and agent
        self.progress_og['value'] = 0
        self.progress_label_og['text'] = "Training Progress: 0%"
        self.plot_learning_curve([], self.ax_og, self.canvas_og_mpl, "Original Grid") # Clear learning curve plot

        try:
            grid_size = self.grid_size_var.get()
            num_obstacles = self.num_obstacles_var.get()

            if not (4 <= grid_size <= 8):
                raise ValueError("Grid size must be between 4 and 8.")
            max_allowed_obstacles = (grid_size * grid_size) // 2
            if not (0 <= num_obstacles <= max_allowed_obstacles):
                raise ValueError(f"Number of obstacles cannot exceed {max_allowed_obstacles} (half of grid boxes).")

            # Get hyperparameters from GUI
            learning_rate = self.lr_var.get()
            discount_factor = self.df_var.get()
            epsilon_decay_rate = self.edr_var.get()
            min_epsilon = self.min_eps_var.get()
            total_episodes = self.episodes_var.get()
            if not (0 < learning_rate <= 1 and 0 <= discount_factor <= 1 and 0 < epsilon_decay_rate <= 1 and 0 <= min_epsilon < 1 and total_episodes > 0):
                raise ValueError("Hyperparameters out of valid range.")

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            return

        # Generate a new grid
        start_state = (0, 0)
        end_state = (grid_size - 1, grid_size - 1)
        obstacles = self.generate_obstacles(grid_size, num_obstacles, start_state, end_state)

        # If generate_obstacles could not find a path, it will return an empty list of obstacles
        # or fewer than requested, but still guarantee a path if possible.
        # The `train_q_learning` function itself has a `has_path` check to confirm.
        if not has_path(grid_size, start_state, end_state, obstacles):
             messagebox.showerror("Grid Generation Error", "Could not generate a solvable grid with the specified number of obstacles. Please reduce obstacle count or increase grid size.")
             return # Abort if no path is guaranteed

        self.current_grid_params = (grid_size, start_state, end_state, obstacles)
        self.draw_grid_on_canvas(grid_size, start_state, end_state, obstacles)
        self.last_drawn_grid_params = self.current_grid_params

        # Run training in a separate thread
        def train_thread():
            self.root.after(0, lambda: messagebox.showinfo("Training Started", "Starting Q-Learning training for Original Grid. GUI will remain responsive."))
            q_table, episode_rewards = train_q_learning(
                grid_size=grid_size,
                start_state=start_state,
                end_state=end_state,
                obstacles=obstacles,
                total_episodes=total_episodes,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon_decay_rate=epsilon_decay_rate,
                min_epsilon=min_epsilon,
                progress_callback=lambda current, total: self.update_progress(current, total, "original")
            )

            if q_table is not None:
                self.current_q_table = q_table
                self.current_tab_trained = "original"
                # Clear real-world map data as we switched mode
                self.current_osm_graph = None
                self.real_world_grid_to_node_map = None
                self.real_world_node_to_grid_map = None
                self.real_world_min_coords = None
                self.real_world_max_coords = None
                self.real_world_graph_start_node = None
                self.real_world_graph_end_node = None

                self.root.after(0, lambda: self.plot_learning_curve(episode_rewards, self.ax_og, self.canvas_og_mpl, "Original Grid"))
                self.root.after(0, lambda: messagebox.showinfo("Training Complete", "Q-Learning training finished for Original Grid. Visualizing path..."))
                self.root.after(0, self.animate_original_grid_path) # Start animation
            else:
                 self.root.after(0, lambda: messagebox.showerror("Training Failed", "Q-Learning training for Original Grid failed or no path found. Check console for details."))
                 self.root.after(0, lambda: self.plot_learning_curve([], self.ax_og, self.canvas_og_mpl, "Original Grid")) # Clear plot


        self.animation_thread = threading.Thread(target=train_thread)
        self.animation_thread.daemon = True # Allow thread to exit with app
        self.animation_thread.start()


    def generate_obstacles(self, grid_size, num_obstacles, start_state, end_state):
        all_possible_cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        available_cells_for_obstacles = [cell for cell in all_possible_cells if cell != start_state and cell != end_state]

        obstacles = set()
        tried_cells = set() # Keep track of cells we've already tried to place an obstacle

        # Prioritize placing obstacles that don't immediately block the path
        # Loop up to a certain number of attempts, or until enough obstacles are placed
        for _ in range(num_obstacles * 5): # Give it more attempts than requested obstacles
            if len(obstacles) >= num_obstacles:
                break
            if not available_cells_for_obstacles:
                print("No more available cells to place obstacles.")
                break

            chosen_obstacle = random.choice(available_cells_for_obstacles)
            if chosen_obstacle in tried_cells: # Avoid retrying the same cell if it constantly blocks
                continue
            tried_cells.add(chosen_obstacle)

            temp_obstacles = list(obstacles | {chosen_obstacle}) # Test with new obstacle added
            if has_path(grid_size, start_state, end_state, temp_obstacles):
                obstacles.add(chosen_obstacle)
                # No need to remove from available_cells_for_obstacles unless we are strict about unique picks,
                # random.choice will eventually pick others. But for efficiency, we can remove.
                available_cells_for_obstacles.remove(chosen_obstacle)
            else:
                # If adding this obstacle blocks the path, don't add it
                # and remove it from available_cells so we don't try it again often
                if chosen_obstacle in available_cells_for_obstacles:
                    available_cells_for_obstacles.remove(chosen_obstacle)

        if len(obstacles) < num_obstacles:
            messagebox.showwarning("Obstacle Placement", f"Could not place all {num_obstacles} obstacles while ensuring a path. Only {len(obstacles)} placed.")
        return list(obstacles)


    def clear_original_grid_canvas(self):
        self.grid_canvas.delete("all")


    def draw_grid_on_canvas(self, grid_size, start_state, end_state, obstacles, current_agent_pos=None, optimal_path=None):
        self.clear_original_grid_canvas()

        canvas_width = self.grid_canvas.winfo_width()
        canvas_height = self.grid_canvas.winfo_height()

        cell_size = min(canvas_width / grid_size, canvas_height / grid_size)
        offset_x = (canvas_width - grid_size * cell_size) / 2
        offset_y = (canvas_height - grid_size * cell_size) / 2

        for r in range(grid_size):
            for c in range(grid_size):
                x1 = offset_x + c * cell_size
                y1 = offset_y + r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                fill_color = "white"
                text = ""
                tag = f"cell_{r}_{c}"

                if (r, c) == start_state:
                    fill_color = "green"
                    text = "S"
                elif (r, c) == end_state:
                    fill_color = "red"
                    text = "E"
                elif (r, c) in obstacles:
                    fill_color = "gray"
                    text = "#"

                self.grid_canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="black", tags=tag)
                if text:
                    self.grid_canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=text, fill="black", tags=tag, font=("Arial", int(cell_size/3)))

        # Draw optimal path if provided (after initial grid to overlay)
        if optimal_path:
            for i, (r, c) in enumerate(optimal_path):
                if (r, c) != start_state and (r, c) != end_state:
                    x_center = offset_x + c * cell_size + cell_size / 2
                    y_center = offset_y + r * cell_size + cell_size / 2
                    radius = cell_size * 0.2
                    self.grid_canvas.create_oval(x_center - radius, y_center - radius,
                                                 x_center + radius, y_center + radius,
                                                 fill="blue", outline="blue", tags="path_marker")

        # Draw current agent position if provided
        if current_agent_pos:
            r, c = current_agent_pos
            x_center = offset_x + c * cell_size + cell_size / 2
            y_center = offset_y + r * cell_size + cell_size / 2
            radius = cell_size * 0.35 # Slightly larger for agent
            # Draw a circle for the agent
            self.grid_canvas.create_oval(x_center - radius, y_center - radius,
                                         x_center + radius, y_center + radius,
                                         fill="gold", outline="black", tags="agent")
        self.grid_canvas.update_idletasks() # Ensure canvas updates


    def animate_original_grid_path(self):
        if self.current_q_table is None or self.current_grid_params is None or self.current_tab_trained != "original":
            messagebox.showwarning("No Data", "Please train the Q-Learning model for the Original Grid first.")
            return

        grid_size, start_state, end_state, obstacles = self.current_grid_params
        optimal_path = find_optimal_path(self.current_q_table, grid_size, start_state, end_state, obstacles)

        if not optimal_path:
            messagebox.showwarning("Path Not Found", "Optimal path could not be determined after training. Agent might be stuck or no path exists.")
            return

        self.root.after(0, lambda: self.draw_grid_on_canvas(grid_size, start_state, end_state, obstacles)) # Redraw base grid

        # Animation loop
        def animate():
            for i, state in enumerate(optimal_path):
                if self.animation_stop_event.is_set():
                    print("Animation stopped by user.")
                    break
                # Only draw agent for current step, not the full path yet
                self.root.after(0, lambda s=state: self.draw_grid_on_canvas(grid_size, start_state, end_state, obstacles, current_agent_pos=s))
                time.sleep(0.2) # Control animation speed

            if not self.animation_stop_event.is_set():
                # After animation, draw the final path clearly and remove agent
                self.root.after(0, lambda: self.draw_grid_on_canvas(grid_size, start_state, end_state, obstacles, optimal_path=optimal_path))
                messagebox.showinfo("Animation Complete", "Agent has finished traversing the optimal path.")

        self.animation_thread = threading.Thread(target=animate)
        self.animation_thread.daemon = True
        self.animation_thread.start()