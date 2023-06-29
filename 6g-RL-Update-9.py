import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the data
df = pd.read_csv('predicted_output.csv')
topology = pd.read_csv('Topology.csv')

# Create a mapping from Link_ID to Device_name
id_to_device = pd.Series(topology.Device_name_1.values, index=topology.Link_ID).to_dict()

# Map Link_ID in df to corresponding Device_name
df.Link_ID = df.Link_ID.map(id_to_device)

# Initialize the Q-table as a dictionary
Q_table = {}

# Parameters
alpha = 0.5  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.1  # Epsilon-greedy strategy for exploration
num_episodes = 100  # Number of training episodes
terminate_early = True  # Terminate if optimal path is found

# Create a graph from the network topology
G = nx.from_pandas_edgelist(topology, 'Device_name_1', 'Device_name_2', edge_attr='Link_ID')

# Every distinct Link_ID and Time pair is a unique state
df['state'] = list(zip(df.Link_ID, df.Time))
states = df['state'].unique()

# Initialize Q-values for all states in Q-table
for state in states:
    Q_table[state] = 0

# For storing the sum of rewards and lengths of optimal paths
rewards = []
path_lengths = []

for episode in tqdm(range(num_episodes), desc="Training"):  # display progress bar
    # Reset the state
    state = df.sample(1)['state'].values[0]
    sum_rewards = 0
    path_length = 0

    while True:  # For each step in the episode
        # Choose an action: either explore or exploit
        if np.random.uniform() < epsilon:  # Exploration
            possible_actions = list(G.neighbors(state[0]))  # Get the possible actions from the network topology
            action = np.random.choice(possible_actions)
        else:  # Exploitation
            action = max(Q_table, key=Q_table.get)[0]

        # Execute the action and observe the reward and next state
        next_state = (action, state[1] + 1)  # Assume the time increases by 1 in each step

        try:
            reward = -abs(df.loc[(df.Link_ID == action) & (df.Time == next_state[1]), 'Current_Moving_Average_throughput'].values[0] - df.loc[(df.Link_ID == action) & (df.Time == next_state[1]), 'Predicted_Moving_Average_throughput'].values[0])
        except IndexError:
            break  # Break the loop if the action is not in the dataframe

        sum_rewards += reward
        path_length += 1

        # Update Q-table using the Q-learning update rule
        Q_table[state] += alpha * (reward + gamma * Q_table[next_state] - Q_table[state])

        state = next_state  # Move to the next state

        # Terminate the episode if optimal path is found
        if terminate_early and state[0] == 'gNB_7':  # Check if the current state is the end node
            break

    rewards.append(sum_rewards)
    path_lengths.append(path_length)

    # Print progress
    if (episode + 1) % 1000 == 0:  # Print progress every 1000 episodes
        print(f"Episode {episode + 1}/{num_episodes} completed.")
        print(f"Size of Q-table: {len(Q_table)}")

        # Print the sum of rewards and length of path for the current episode
        print(f"Sum of rewards: {sum_rewards}")
        print(f"Length of path: {path_length}")

# Print the different values for each episode
print("Different values for each episode:")
for episode in range(len(rewards)):
    print(f"Episode {episode+1}: Sum of rewards = {rewards[episode]}, Length of path = {path_lengths[episode]}")

# The optimal routing path is the one with maximum Q-value
optimal_path = nx.shortest_path(G, 'gNB_18', 'gNB_8', weight='Link_ID')

# Calculate the total weight or predicted throughput of the optimal path
total_weight = sum(1 / G[optimal_path[i]][optimal_path[i + 1]]['Link_ID'] for i in range(len(optimal_path) - 1))
print("Total weight:", total_weight)

# Print the weight for each link in the optimal path
for i in range(len(optimal_path) - 1):
    link_weight = 1 / G[optimal_path[i]][optimal_path[i + 1]]['Link_ID']
    print(f"Link: {optimal_path[i]} -> {optimal_path[i + 1]}, Weight: {link_weight}")

# Convert the optimal path to a list of nodes
optimal_path_nodes = [node for node in optimal_path if isinstance(node, str)]

# Visualization of network topology and optimal path
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos=pos, with_labels=True, node_color='skyblue')

# Highlight the optimal path in red color
optimal_path_edges = [(optimal_path_nodes[i], optimal_path_nodes[i + 1]) for i in
                          range(len(optimal_path_nodes) - 1)]
nx.draw_networkx_edges(G, pos=pos, edgelist=optimal_path_edges, edge_color='r', width=2)

plt.title('Network Topology with Optimal Path')
# Save the figure in a PDF file before showing it
plt.savefig('Network_Topology_opt_path.pdf', format='pdf')
plt.show()

# Plot the sum of rewards and path lengths over episodes
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Sum of rewards')
plt.title('Sum of rewards per episode')

plt.subplot(1, 2, 2)
plt.plot(path_lengths)
plt.xlabel('Episode')
plt.ylabel('Path length')
plt.title('Length of path per episode')

plt.tight_layout()
plt.show()

# Print average path length and reward
average_reward = np.mean(rewards)
average_path_length = np.mean(path_lengths)
print(f"Average reward: {average_reward}")
print(f"Average path length: {average_path_length}")

