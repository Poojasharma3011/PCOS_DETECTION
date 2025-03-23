import numpy as np
import pandas as pd
import gym
from gym import spaces
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("PCOS_DATASET_AUGMENTED_WITH_BMI.csv")

# Select features and target
features = ['Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Blood Group', 'Cycle(R/I)', 'Cycle length(days)', 'Weight gain(Y/N)',
             'hair growth(Y/N)','Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)',
            'Reg.Exercise(Y/N)', 'Follicle No. (R)', 'Follicle No. (L)', 'TSH (mIU/L)']
target = 'PCOS (Y/N)'

df = df[features + [target]].dropna()
X = df[features].values
y = df[target].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define custom PCOS detection environment
class PCOSDetectionEnv(gym.Env):
    def __init__(self, X, y):
        super(PCOSDetectionEnv, self).__init__()
        self.X = X
        self.y = y
        self.index = 0
        self.observation_space = spaces.Box(low=-3, high=3, shape=(X.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0: No PCOS, 1: PCOS

    def reset(self):
        self.index = 0
        return self.X[self.index]

    def step(self, action):
        correct = int(action == self.y[self.index])
        reward = 1 if correct else -1
        self.index += 1
        done = self.index >= len(self.X)
        return (self.X[self.index] if not done else np.zeros(self.X.shape[1]), reward, done, {})

# Initialize environment
env = PCOSDetectionEnv(X_train, y_train)

# Q-learning setup
q_table = np.zeros((len(X_train), 2))
alpha, gamma, epsilon = 0.1, 0.9, 0.1  # Learning rate, discount factor, exploration-exploitation balance

def train_q_learning(env, q_table, episodes=1000):
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            state_idx = env.index
            action = np.argmax(q_table[state_idx]) if np.random.rand() >= epsilon else env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            next_state_idx = env.index if not done else state_idx
            q_table[state_idx, action] += alpha * (reward + gamma * np.max(q_table[next_state_idx]) - q_table[state_idx, action])

train_q_learning(env, q_table)

# Save the model
with open("pcos_model.pkl", "wb") as file:
    pickle.dump((q_table, X_train), file)