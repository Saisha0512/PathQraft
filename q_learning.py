import collections
import random
import numpy as np
from tkinter import messagebox

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]


NUM_ACTIONS = len(ACTIONS)


GOAL_REWARD = 100


OBSTACLE_REWARD = -10


DEFAULT_REWARD = -1


def is_valid_state(state, grid_size, obstacles):
    row, col = state
    return 0 <= row < grid_size and 0 <= col < grid_size and state not in obstacles


def get_next_state(current_state, action_index, grid_size, obstacles):
    row, col = current_state
    dr, dc = ACTIONS[action_index]
    next_row, next_col = row + dr, col + dc
    next_state = (next_row, next_col)

    if is_valid_state(next_state, grid_size, obstacles):
        return next_state
    else:
        return current_state # Stay in the current state if the move is invalid (hit boundary or obstacle)


def get_reward(state, end_state, obstacles):
    if state == end_state:
        return GOAL_REWARD
    elif state in obstacles:
        return OBSTACLE_REWARD
    else:
        return DEFAULT_REWARD


def train_q_learning(grid_size, start_state, end_state, obstacles, total_episodes, learning_rate, discount_factor, epsilon_decay_rate, min_epsilon, progress_callback=None):
    """Runs the Q-Learning training process."""
    if not grid_size or start_state is None or end_state is None or obstacles is None:
        print("Cannot train: Invalid grid parameters.")
        return None, []

    q_table = np.zeros((grid_size, grid_size, NUM_ACTIONS))
    epsilon = 1.0 # Epsilon starts high for exploration
    episode_rewards = []

    print(f"Starting Q-Learning training on {grid_size}x{grid_size} grid...")

    # Check if start or end state is an obstacle
    if start_state in obstacles or end_state in obstacles:
        print("Start or End state is an obstacle. Training aborted.")
        messagebox.showerror("Training Error", "Start or End position cannot be an obstacle. Please regenerate grid or select different map start/end points.")
        return None, []

    # Check if a path exists using BFS before training
    if not has_path(grid_size, start_state, end_state, obstacles):
        print("No path exists between start and end with given obstacles. Training aborted.")
        messagebox.showerror("Training Error", "No possible path exists between start and end with current obstacles. Please regenerate grid or reduce obstacles for Original Grid, or try a different location/grid size for Real-World Map.")
        return None, []

    for episode in range(total_episodes):
        current_state = start_state
        done = False
        total_episode_reward = 0
        step_count = 0
        max_steps_per_episode = grid_size * grid_size * 4 # Prevent infinite loops in difficult grids

        while not done and step_count < max_steps_per_episode:
            row, col = current_state
            if random.uniform(0, 1) < epsilon:
                action_index = random.randint(0, NUM_ACTIONS - 1)
            else:
                q_values = q_table[row, col]
                # Handle cases where all Q-values are the same (e.g., all zeros initially)
                max_q = np.max(q_values)
                best_actions = np.where(q_values == max_q)[0]
                action_index = random.choice(best_actions)

            next_state = get_next_state(current_state, action_index, grid_size, obstacles)
            reward = get_reward(next_state, end_state, obstacles)
            total_episode_reward += reward

            next_row, next_col = next_state
            old_value = q_table[row, col, action_index]
            next_max_value = np.max(q_table[next_row, next_col])
            new_value = old_value + learning_rate * (reward + discount_factor * next_max_value - old_value)
            q_table[row, col, action_index] = new_value

            current_state = next_state

            if current_state == end_state:
                done = True
            step_count += 1

        epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)
        episode_rewards.append(total_episode_reward)

        if progress_callback:
            progress_callback(episode + 1, total_episodes)

    print("Training finished.")
    return q_table, episode_rewards


def find_optimal_path(q_table, grid_size, start_state, end_state, obstacles):
    """Finds the optimal path using the trained Q-table."""
    if q_table is None or grid_size is None or start_state is None or end_state is None or obstacles is None:
        print("Cannot find path: Invalid input.")
        return []

    print("\nFinding optimal path using the trained Q-table:")
    current_state = start_state
    optimal_path = []
    step_count = 0
    max_path_length = grid_size * grid_size * 2 # Prevent infinite loops in pathfinding

    # Add start state to path
    optimal_path.append(current_state)

    while current_state != end_state and step_count < max_path_length:
        row, col = current_state
        if not (0 <= row < grid_size and 0 <= col < grid_size):
            print(f"Agent went out of bounds at {current_state}. Pathfinding stopped.")
            break # Safety break if state becomes invalid

        # If current_state somehow became an obstacle, break
        if (row, col) in obstacles:
            print(f"Agent stepped into an obstacle at {current_state}. Pathfinding stopped.")
            break

        # Check if all Q-values are identical (e.g., all zeros or very small)
        q_values_for_state = q_table[row, col]
        if np.all(q_values_for_state == q_values_for_state[0]):
            # If all Q-values are the same, pick a random valid action
            valid_actions_indices = []
            for i, action in enumerate(ACTIONS):
                dr, dc = action
                next_row, next_col = row + dr, col + dc
                if 0 <= next_row < grid_size and 0 <= next_col < grid_size and (next_row, next_col) not in obstacles:
                    valid_actions_indices.append(i)
            if valid_actions_indices:
                action_index = random.choice(valid_actions_indices)
            else:
                # Agent is completely blocked from moving anywhere from this state
                print(f"Agent is trapped at {current_state}. Cannot reach goal.")
                break
        else:
            action_index = np.argmax(q_values_for_state)

        next_state = get_next_state(current_state, action_index, grid_size, obstacles)

        # If agent cannot move to a new state and is not at the end state, it's stuck.
        if next_state == current_state and current_state != end_state:
            print(f"Agent stuck at {current_state}. Cannot reach goal.")
            break

        current_state = next_state
        optimal_path.append(current_state)
        step_count += 1

    if current_state == end_state:
        print("Goal reached.")
    else:
        print("Pathfinding terminated without reaching goal (might be stuck or path too long).")

    return optimal_path


def has_path(grid_size, start, end, obstacles):
    """Checks if a path exists from start to end using BFS."""
    queue = collections.deque([start])
    visited = {start}

    while queue:
        r, c = queue.popleft()

        if (r, c) == end:
            return True

        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            next_state = (nr, nc)

            if 0 <= nr < grid_size and 0 <= nc < grid_size and \
               next_state not in obstacles and next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)
    return False