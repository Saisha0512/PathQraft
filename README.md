# PathQraft – Intelligent Pathfinding Using Q-Learning

> **Intelligent Pathfinding Through Reinforcement Learning**

PathQraft is a Python-based pathfinding simulator that demonstrates the application of **Reinforcement Learning (Q-Learning)** to discover optimal paths in both grid-based and real-world environments. It provides an interactive graphical interface where users can configure learning parameters, train an intelligent agent, and visualize its learning process and navigation decisions.

---

## Disclaimer & Copyright Notice

This project is the intellectual property of its original author.

**All rights reserved.** The code is strictly prohibited from being used, copied, modified, distributed, reproduced, or published in any form without the explicit written permission of the original author.

---

## Project Overview

PathQraft explores how an intelligent agent can learn to navigate an environment through trial and error using Q-Learning.

The project provides two simulation environments:

- **Grid-Based Pathfinding:** An agent learns to navigate a customizable grid while avoiding obstacles and reaching its destination.
- **Real-World Pathfinding:** An agent learns to navigate real-world road networks using geographical data obtained through OpenStreetMap.

Through an interactive Tkinter GUI, users can configure training parameters, execute learning episodes, and visualize the agent's progress and learned paths.

---

## Key Features

- **Customizable Grid Environment** – Configure grid dimensions and obstacles.
- **Real-World Map Integration** – Retrieve and visualize geographical road networks using OpenStreetMap and OSMnx.
- **Q-Learning Algorithm** – Train an agent to learn optimal navigation strategies.
- **Learning Curve Visualization** – Observe the agent's learning progress during training.
- **Optimal Path Visualization** – Display the learned path from the starting point to the destination.
- **Agent Animation** – Visualize the agent navigating through the environment.
- **Adjustable Hyperparameters** – Experiment with learning rate, discount factor, exploration decay, and training episodes.
- **Utility Tools** – Access additional utilities through the graphical interface.

---

## How It Works

PathQraft uses **Q-Learning**, a model-free Reinforcement Learning algorithm, to help an agent learn the best actions to take in an environment.

The agent interacts with the environment, receives rewards or penalties, and updates its Q-values based on its experiences.

### Q-Learning Update Equation

The Q-value is updated using the following equation:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'}Q(s',a') - Q(s,a)\right]
\]

Where:

- **Q(s, a):** Current Q-value for taking action `a` in state `s`.
- **α (Learning Rate):** Controls how much newly acquired information influences the existing Q-value.
- **r (Reward):** Immediate reward received after taking an action.
- **γ (Discount Factor):** Determines the importance of future rewards.
- **s':** Next state reached by the agent.
- **a':** Possible action in the next state.

Through repeated training episodes, the agent gradually improves its navigation strategy.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.x | Core programming language |
| Tkinter | Graphical User Interface |
| NumPy | Numerical computations and Q-table operations |
| Matplotlib | Learning curves and visualizations |
| NetworkX | Graph representation and network operations |
| OSMnx | Real-world geographical and road network data |
| GeoPandas | Geospatial data handling |
| Pillow (PIL) | Image processing and GUI image support |
| Threading | Background execution and animations |
| OpenStreetMap | Real-world map data source |

---

## Project Structure

The project is organized into multiple modules to improve readability, maintainability, and code organization.

```text
PathQraft/
│
├── main.py
├── app.py
├── dependencies.py
├── utilities.py
├── q_learning.py
├── map_utils.py
│
├── ui/
│   ├── __init__.py
│   ├── menu.py
│   ├── utilities.py
│   ├── grid.py
│   └── real_world.py
│
├── requirements.txt
└── README.md
```

### Module Description

| File / Directory | Description |
|---|---|
| `main.py` | Application entry point; checks dependencies and launches the application. |
| `app.py` | Main application class that combines the GUI modules. |
| `dependencies.py` | Handles dependency checking and installation. |
| `utilities.py` | Contains common utility functions. |
| `q_learning.py` | Contains the Q-Learning implementation and related logic. |
| `map_utils.py` | Handles map-related and geographical operations. |
| `ui/menu.py` | Implements the main menu and navigation interface. |
| `ui/utilities.py` | Implements the utility tools interface. |
| `ui/grid.py` | Handles grid-based training, visualization, and animation. |
| `ui/real_world.py` | Handles real-world map training, visualization, and animation. |
| `requirements.txt` | Lists the project's Python dependencies. |

---

## Getting Started

Follow these steps to set up and run PathQraft locally.

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
```

### 2. Navigate to the Project Directory

```bash
cd PathQraft
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python main.py
```

The graphical application will launch, allowing you to explore the available pathfinding simulations.

---

## Configurable Hyperparameters

PathQraft allows users to experiment with different Q-Learning parameters through its graphical interface.

| Hyperparameter | Description |
|---|---|
| **Learning Rate (α)** | Controls how strongly new information updates existing Q-values. |
| **Discount Factor (γ)** | Determines the importance of future rewards. |
| **Epsilon Decay Rate** | Controls how quickly exploration decreases during training. |
| **Minimum Epsilon** | Defines the minimum exploration probability. |
| **Total Episodes** | Specifies the number of training iterations. |

---

## Simulation Environments

### 1. Grid-Based Pathfinding

The grid environment provides a controlled setting for experimenting with Reinforcement Learning.

- Select the grid dimensions.
- Configure the number of obstacles.
- Train the agent using Q-Learning.
- Observe the agent's learning progress.
- Visualize its learned path through the grid.

### 2. Real-World Pathfinding

The real-world environment extends the pathfinding simulation to geographical road networks.

- Enter a geographical location (e.g., Mumbai, India).
- Alternatively, use the random location functionality.
- Retrieve road network data using OpenStreetMap and OSMnx.
- Represent the geographical network as a graph.
- Train the agent to navigate the selected network.
- Visualize the resulting navigation path.

### 3. Utility Tools

The application also provides additional utilities through its graphical interface, including:

- Viewing the current working directory.
- Simulating file uploads.
- Removing selected characters from strings.

---

## Project Objectives

- Understand and implement Reinforcement Learning using Q-Learning.
- Demonstrate how an agent learns through interaction with its environment.
- Apply the same learning approach to both simulated and geographical environments.
- Visualize the relationship between training episodes and learning performance.
- Explore the practical application of intelligent pathfinding algorithms.

---

## Acknowledgements

This project was developed as an academic demonstration of Reinforcement Learning and intelligent pathfinding.

Special thanks to the developers and communities behind the open-source technologies used in this project, particularly:

- Python
- OpenStreetMap
- OSMnx
- NumPy
- Matplotlib
- NetworkX
- GeoPandas

---

_**PathQraft – Learning to Find the Way.**_
