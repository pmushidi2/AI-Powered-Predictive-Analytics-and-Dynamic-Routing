import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv("Dataset.csv")

# Load the predicted output
predicted_output = pd.read_csv("predicted_output.csv")

# Load the network topology
topology = pd.read_csv("Topology.csv")

# Define the state space
state_features = ["Link_ID", "Time", "Moving_Average_throughput", "Instantaneous_throughput", "Time_average_throughput"]
state_data = data[state_features].values

# Define the action space
action_space = topology[["Device_name_1", "Device_name_2", "Link_ID"]]

# Define the rewards
def calculate_reward(state, action):
    link_id = state[0]  # Assuming the "Link_ID" is the first element in the state array
    selected_link = action["Link_ID"]
    selected_link_throughput = data[(data["Link_ID"] == link_id) & (data["Link_ID"] == selected_link)]

    if selected_link_throughput.empty or "Predicted_Moving_Average_throughput" not in selected_link_throughput.columns:
        return 0  # Return a default reward value when the selected link throughput is not available or the column doesn't exist

    selected_link_predicted_throughput = selected_link_throughput["Predicted_Moving_Average_throughput"].values[0]
    reward = selected_link_predicted_throughput

    return reward

# Define the Q-Learning algorithm
num_states = len(state_data)
num_actions = len(action_space)

# Initialize the q_table with the correct shape
q_table = np.zeros((num_states, num_actions))

# Check if the size of the action space matches the size of the second axis of the q_table
if num_actions != q_table.shape[1]:
    raise ValueError("The size of the action space does not match the size of the q_table.")

alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration vs. exploitation trade-off

# Define stopping criterion
max_episodes = 50  # Maximum number of training episodes
# Training the RL agent
convergence_threshold = 0.001  # Convergence threshold for Q-values

for episode in range(max_episodes):  # Number of training episodes
    print("Episode:", episode)
    state = state_data[0]  # Starting state

    while True:
        # Choose action based on epsilon-greedy strategy
        if np.random.uniform(0, 1) < epsilon:
            action_index = np.random.choice(range(num_actions))
            action = action_space.iloc[action_index]
        else:
            state_index = np.where((state_data == state).all(axis=1))[0][0]
            action_index = np.argmax(q_table[state_index])
            action = action_space.iloc[action_index]

        # Calculate reward
        reward = calculate_reward(state, action)

        # Get next state
        next_state_index = np.random.choice(num_states)  # Random next state for demonstration purposes
        next_state = state_data[next_state_index]

        # Update Q-value
        state_index = np.where((state_data == state).all(axis=1))[0][0]
        next_state_index = np.where((state_data == next_state).all(axis=1))[0][0]

        if 0 <= action_index < num_actions:
            max_q = np.max(q_table[next_state_index])
            q_table[state_index, action_index] = (1 - alpha) * q_table[state_index, action_index] + alpha * (
                        reward + gamma * max_q)
            state_index = np.where((state_data == state).all(axis=1))[0][0]
            next_state_index = np.where((state_data == next_state).all(axis=1))[0][0]

        if 0 <= action_index < num_actions:
            max_q = np.max(q_table[next_state_index])
            q_table[state_index, action_index] = (1 - alpha) * q_table[state_index, action_index] + alpha * (
                        reward + gamma * max_q)
        else:
            print(f"Ignoring invalid action index: {action_index}")
            print(f"State index: {state_index}")

        print(state)
        state = next_state

        # Check convergence of Q-values
        if np.abs(q_table[state_index, action_index] - (reward + gamma * max_q)) < convergence_threshold:
            break

    if episode == max_episodes - 1:  # For the last episode
        # Update routing decisions based on the optimal policy
        optimal_actions = np.argmax(q_table, axis=1)
        repeated_actions = np.repeat(optimal_actions, len(action_space) // len(optimal_actions))
        if len(repeated_actions) == 0:
            repeated_actions = np.zeros(len(action_space), dtype=int)
        else:
            repeated_actions = repeated_actions[:len(action_space)]
        print("Length of repeated_actions:", len(repeated_actions))  # Added line for debugging
        action_space["Optimal_Routing"] = action_space["Link_ID"].values[repeated_actions]

        # Merge with the initial network topology
        output = pd.merge(topology, action_space, on=["Device_name_1", "Device_name_2", "Link_ID"], how="left")

        # Save the updated output to "output.csv"
        output.to_csv("output.csv", index=False)
        print("Updated routing decisions saved to output.csv")










