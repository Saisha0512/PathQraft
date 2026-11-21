import random
import numpy as np
import osmnx as ox
from tkinter import messagebox


def get_real_world_graph(place_name="Random Location"):
    """Downloads a street network graph."""
    try:
        print(f"Attempting to download graph for: {place_name}")
        if place_name.lower() == "random location":
            # Example coordinates for a small area if random is too broad
            # These are for a small part of San Francisco for reproducibility/smaller download
            lat, lon = 37.7849, -122.4172 # Near Civic Center, San Francisco
            G = ox.graph_from_point((lat, lon), dist=500, network_type='drive')
            print(f"Downloaded graph for a random location near ({lat:.4f}, {lon:.4f})")
        else:
            G = ox.graph_from_place(place_name, network_type='drive')
            print(f"Downloaded graph for {place_name}")
        return G
    except Exception as e:
        print(f"Error downloading graph: {e}")
        messagebox.showerror("Graph Download Error", f"Could not download graph for '{place_name}'. Common issues: incorrect place name, no internet, or OSMnx server issues.\nError: {e}")
        return None


def get_start_end_nodes(graph):
    """Selects random start and end nodes from the graph."""
    if not graph or not graph.nodes():
        print("Graph is empty or invalid.")
        return None, None
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        print("Graph has too few nodes to select start/end.")
        return None, None
    start_node = random.choice(nodes)
    end_node = random.choice(nodes)
    while start_node == end_node:
        end_node = random.choice(nodes)
    print(f"Selected graph start node: {start_node}, End node: {end_node}")
    return start_node, end_node


def graph_to_simplified_grid(graph, start_node_graph, end_node_graph, grid_size=30):
    """
    Converts a portion of the graph into a simplified grid representation for Q-learning.
    Returns: grid_size, start_grid_state, end_grid_state, grid_obstacles, grid_to_node_map,
             node_to_grid_map, min_coords, max_coords
    """
    if not graph or start_node_graph is None or end_node_graph is None:
        print("Graph or start/end nodes invalid for grid conversion.")
        return None, None, None, None, None, None, None, None

    nodes_data = {node: data for node, data in graph.nodes(data=True)}

    if start_node_graph not in nodes_data or end_node_graph not in nodes_data:
        print("Start or end node not found in graph data. This shouldn't happen if selected from graph.")
        messagebox.showerror("Graph Data Error", "Internal error: Start or end node not found in graph data.")
        return None, None, None, None, None, None, None, None

    coords = np.array([(data['x'], data['y']) for node, data in nodes_data.items()])

    if len(coords) == 0:
        print("No node coordinates found in the graph.")
        messagebox.showerror("Graph Error", "No node coordinates found in the graph. Cannot create a grid.")
        return None, None, None, None, None, None, None, None

    min_x, min_y = np.min(coords, axis=0)
    max_x, max_y = np.max(coords, axis=0)

    # Handle cases where graph has no spatial extent (e.g., all nodes at same point)
    # Add a small buffer to ranges to prevent division by zero or very small ranges
    range_x = max_x - min_x
    range_y = max_y - min_y
    if range_x < 1e-6: range_x = 1e-6 # Minimum range to prevent division by zero
    if range_y < 1e-6: range_y = 1e-6

    node_to_grid_map = {}
    grid_to_node_map = {} # Map grid cells back to graph nodes for path reconstruction

    start_grid_state = None
    end_grid_state = None

    def map_coord_to_grid(x, y, min_x_val, min_y_val, max_x_val, max_y_val, grid_dim, x_range, y_range):
        # Normalize coordinates to [0, 1) and then scale to grid_dim
        # Invert y-axis mapping for grid to match visual (row 0 at top, higher latitude)
        row = int(((max_y_val - y) / y_range) * grid_dim)
        col = int(((x - min_x_val) / x_range) * grid_dim)

        # Ensure coordinates stay within grid bounds [0, grid_dim-1]
        row = max(0, min(grid_dim - 1, row))
        col = max(0, min(grid_dim - 1, col))
        return row, col

    for node, data in nodes_data.items():
        row, col = map_coord_to_grid(data['x'], data['y'], min_x, min_y, max_x, max_y, grid_size, range_x, range_y)
        grid_state = (row, col)

        # Associate this grid state with the graph node
        node_to_grid_map[node] = grid_state
        # Map grid cell back to a graph node (use the last one for simplicity if multiple nodes map to same cell)
        # This will be useful for converting grid path back to graph path
        grid_to_node_map.setdefault(grid_state, node) # Only set if not already set, keeping the first node mapped to this cell

    # Identify grid cells that are not mapped to any node as potential obstacles
    all_grid_cells = {(r, c) for r in range(grid_size) for c in range(grid_size)}
    mapped_grid_cells = set(grid_to_node_map.keys())
    grid_obstacles = all_grid_cells - mapped_grid_cells

    # Convert start and end graph nodes to grid states
    start_grid_state = node_to_grid_map.get(start_node_graph)
    end_grid_state = node_to_grid_map.get(end_node_graph)

    # It's possible that the initial mapping of start/end graph nodes landed them on an obstacle
    # or outside the valid grid (due to rounding/edges). Find nearest valid grid cell if so.
    if start_grid_state in grid_obstacles:
        temp_start_grid_state = find_nearest_mapped_cell(start_grid_state, mapped_grid_cells)
        if temp_start_grid_state:
            start_grid_state = temp_start_grid_state
            # Update the corresponding graph node in grid_to_node_map for consistency
            start_node_graph = grid_to_node_map[start_grid_state] # Update the graph node too
        else:
            print("Could not find a valid start grid state after finding nearest. Aborting grid creation.")
            messagebox.showerror("Grid Error", "Could not find a valid start grid state for the map. Try a different location or grid size.")
            return None, None, None, None, None, None, None, None

    if end_grid_state in grid_obstacles:
        temp_end_grid_state = find_nearest_mapped_cell(end_grid_state, mapped_grid_cells)
        if temp_end_grid_state:
            end_grid_state = temp_end_grid_state
            # Update the corresponding graph node in grid_to_node_map for consistency
            end_node_graph = grid_to_node_map[end_grid_state] # Update the graph node too
        else:
            print("Could not find a valid end grid state after finding nearest. Aborting grid creation.")
            messagebox.showerror("Grid Error", "Could not find a valid end grid state for the map. Try a different location or grid size.")
            return None, None, None, None, None, None, None, None

    if start_grid_state == end_grid_state:
        print(f"Start and end nodes (or their nearest mapped cells) mapped to the same grid cell: {start_grid_state}. Aborting grid creation.")
        messagebox.showerror("Grid Error", f"Start and end points map to the same grid cell ({start_grid_state}). Please try a different location or grid size.")
        return None, None, None, None, None, None, None, None

    # Ensure start and end are not in obstacles *after* any adjustments
    grid_obstacles.discard(start_grid_state)
    grid_obstacles.discard(end_grid_state)

    print(f"Simplified grid size: {grid_size}x{grid_size}")
    print(f"Start grid state: {start_grid_state}")
    print(f"End grid state: {end_grid_state}")
    print(f"Number of obstacles in grid: {len(grid_obstacles)}")

    return grid_size, start_grid_state, end_grid_state, list(grid_obstacles), grid_to_node_map, node_to_grid_map, (min_x, min_y), (max_x, max_y), start_node_graph, end_node_graph


def find_nearest_mapped_cell(target_cell, mapped_cells):
    """Finds the nearest cell in mapped_cells to the target_cell."""
    if not mapped_cells:
        return None
    target_row, target_col = target_cell
    min_dist_sq = float('inf')
    nearest_cell = None

    for cell in mapped_cells:
        row, col = cell
        dist_sq = (row - target_row)**2 + (col - target_col)**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest_cell = cell
    return nearest_cell