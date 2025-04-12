import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.metrics import root_mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random

# -- Use GPU if available --
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Read & Analyze Dataset --
data = pd.read_csv("RecSys_torch/ml-latest-small/ratings.csv")
data.info()
print(f"Number of unique users: {data.userId.nunique()}")
print(f"Number of unique movies: {data.movieId.nunique()}")
print(f"Distribution of ratings: {data.rating.value_counts()}")
print(f"Dataset size: {data.shape}")

# -- Define Movie Dataset --
class MovieDataset:
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, item):
        users = self.users[item]
        movies = self.movies[item]
        ratings = self.ratings[item]

        return{
            "users": torch.tensor(users, dtype=torch.long),
            "movies": torch.tensor(movies, dtype=torch.long),
            "ratings": torch.tensor(ratings, dtype=torch.float)
        }

# -- DQN Model for Recommendation --
class DQNRecSysModel(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=32):
        super(DQNRecSysModel, self).__init__()
        
        self.n_movies = n_movies
        self.user_embed = nn.Embedding(n_users, embedding_dim)
        self.movie_embed = nn.Embedding(n_movies, embedding_dim)
        
        # State representation network
        self.state_net = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Action-value network (Q-network)
        self.value_net = nn.Sequential(
            nn.Linear(32 + embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, users, movies=None):
        users = users.to(device)
        
        # Get user embeddings as the state
        user_embeds = self.user_embed(users)
        state_value = self.state_net(user_embeds)
        
        if movies is not None:
            # For specific user-movie pairs
            movies = movies.to(device)
            movie_embeds = self.movie_embed(movies)
            
            # Handle case where users and movies have different batch sizes
            if user_embeds.size(0) != movie_embeds.size(0):
                # If only one user but multiple movies
                if user_embeds.size(0) == 1 and movie_embeds.size(0) > 1:
                    state_value = state_value.repeat(movie_embeds.size(0), 1)
                # If only one movie but multiple users
                elif movie_embeds.size(0) == 1 and user_embeds.size(0) > 1:
                    movie_embeds = movie_embeds.repeat(user_embeds.size(0), 1)
                else:
                    # Something's wrong with the sizes
                    raise ValueError(f"Incompatible batch sizes: users {user_embeds.size(0)}, movies {movie_embeds.size(0)}")
            
            # Combine state and action
            combined = torch.cat([state_value, movie_embeds], dim=1)
            q_value = self.value_net(combined)
            return q_value
        else:
            # For all possible movies (used during recommendation)
            # Process in batches to avoid memory issues
            q_values = []
            batch_size = users.shape[0]
            
            # Process 100 movies at a time to avoid memory issues
            movie_batch_size = 100
            for i in range(0, self.n_movies, movie_batch_size):
                end_idx = min(i + movie_batch_size, self.n_movies)
                batch_q_values = []
                
                for j in range(i, end_idx):
                    movie_batch = torch.full((batch_size,), j, dtype=torch.long).to(device)
                    movie_embeds = self.movie_embed(movie_batch)
                    combined = torch.cat([state_value, movie_embeds], dim=1)
                    q_value = self.value_net(combined)
                    batch_q_values.append(q_value)
                
                batch_q = torch.cat(batch_q_values, dim=1)
                q_values.append(batch_q)
            
            return torch.cat(q_values, dim=1)

# -- Memory Replay Buffer --
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# -- RL Agent --
class DQNAgent:
    def __init__(self, state_size, action_size, model, target_model, replay_buffer, 
                 batch_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.model = model
        self.target_model = target_model
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def select_action(self, state, available_actions=None):
        if available_actions is None:
            available_actions = list(range(self.action_size))
            
        # Epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)
        
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.long).to(device)
            q_values = self.model(state_tensor)
            
            # Mask unavailable actions with a large negative value
            if len(available_actions) < self.action_size:
                mask = torch.ones(self.action_size) * -1e9
                mask[available_actions] = 0
                q_values += mask.to(device)
                
            return torch.argmax(q_values).item()
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.long).to(device)
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(device)
        next_states = torch.tensor(next_states, dtype=torch.long).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1).to(device)
        
        # Compute current Q values for the specific actions taken
        self.model.train()
        movies = actions.squeeze()  # Get the specific movies that were chosen
        curr_q_values = self.model(states, movies)
        
        # Compute target Q values
        self.target_model.eval()
        with torch.no_grad():
            # For each next state, evaluate all possible actions and take max
            max_next_q_values = []
            for i, next_state in enumerate(next_states):
                next_state_tensor = next_state.unsqueeze(0)  # Add batch dimension
                q_values_for_all_movies = []
                
                # We need to evaluate all movies in smaller batches to avoid OOM
                batch_size = 100  # Process 100 movies at a time
                for start_idx in range(0, self.action_size, batch_size):
                    end_idx = min(start_idx + batch_size, self.action_size)
                    movie_batch = torch.arange(start_idx, end_idx, dtype=torch.long).to(device)
                    # Repeat the same state for each movie
                    state_batch = next_state_tensor.repeat(end_idx - start_idx)
                    q_values = self.target_model(state_batch, movie_batch)
                    q_values_for_all_movies.append(q_values)
                
                all_q_values = torch.cat(q_values_for_all_movies)
                max_q = all_q_values.max().item()
                max_next_q_values.append(max_q)
            
            max_next_q = torch.tensor(max_next_q_values, dtype=torch.float32).view(-1, 1).to(device)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Update model
        loss = self.loss_fn(curr_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

# -- Data Preprocessing --
label_user = preprocessing.LabelEncoder()
label_movie = preprocessing.LabelEncoder()
data.userId = label_user.fit_transform(data.userId.values)
data.movieId = label_movie.fit_transform(data.movieId.values)

# -- Split into train and validation datasets --
data_train, data_valid = model_selection.train_test_split(
    data, test_size=0.2, random_state=42, stratify=data.rating.values
)

# -- Create environment --
class MovieEnv:
    def __init__(self, data, n_users, n_movies):
        self.data = data
        self.n_users = n_users
        self.n_movies = n_movies
        self.user_history = defaultdict(list)
        
        # Create a mapping of (user_id, movie_id) to rating
        self.ratings = {}
        for _, row in data.iterrows():
            self.ratings[(row['userId'], row['movieId'])] = row['rating']
            self.user_history[row['userId']].append(row['movieId'])
    
    def reset(self, user_id=None):
        if user_id is None:
            # Pick a random user
            user_id = np.random.randint(0, self.n_users)
        
        self.current_user = user_id
        # Return user_id as state
        return user_id
    
    def step(self, movie_id):
        # Calculate reward based on actual rating if available
        rating_key = (self.current_user, movie_id)
        if rating_key in self.ratings:
            # Convert rating scale (1-5) to reward (-1 to 1)
            rating = self.ratings[rating_key]
            reward = (rating - 3) / 2  # Maps 1->-1, 3->0, 5->1
        else:
            # No rating available, assume neutral
            reward = 0
            
        # In a full implementation, you would update user state here
        # For simplicity, we'll keep the same user state
        next_state = self.current_user
        done = False  # In a real scenario, you'd set this based on session end
        
        return next_state, reward, done
    
    def get_available_actions(self, user_id):
        # Return movies the user hasn't rated yet
        rated_movies = set(self.user_history[user_id])
        all_movies = set(range(self.n_movies))
        return list(all_movies - rated_movies)

# -- Create models, environment, and agent --
n_users = len(label_user.classes_)
n_movies = len(label_movie.classes_)

# Create main model and target model
main_model = DQNRecSysModel(n_users=n_users, n_movies=n_movies).to(device)
target_model = DQNRecSysModel(n_users=n_users, n_movies=n_movies).to(device)
target_model.load_state_dict(main_model.state_dict())

# Create replay buffer
replay_buffer = ReplayBuffer(capacity=50000)

# Create environment
env = MovieEnv(data_train, n_users, n_movies)

# Create agent
agent = DQNAgent(
    state_size=1,  # user_id
    action_size=n_movies,
    model=main_model,
    target_model=target_model,
    replay_buffer=replay_buffer,
    batch_size=64,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.1,
    epsilon_decay=0.995
)

# -- Training Loop --
n_episodes = 500
update_target_every = 10
print_every = 100 #Print per episodes
total_reward = 0
losses = []

writer = SummaryWriter()

print(f"\nTraining on: {device}")
print("Starting training loop...")
for episode in range(n_episodes):
    # Pick a random user
    user_id = np.random.randint(0, n_users)
    state = env.reset(user_id)
    
    # Get available movies for this user
    available_actions = env.get_available_actions(user_id)
    
    if not available_actions:
        continue  # Skip users who have rated all movies
    
    # Select an action (movie to recommend)
    action = agent.select_action(state, available_actions)
    
    # Take the action and observe next state and reward
    next_state, reward, done = env.step(action)
    
    # Store in replay buffer
    replay_buffer.add(state, action, reward, next_state, done)
    
    # Train the agent
    loss = agent.train()
    if loss > 0:
        losses.append(loss)
    
    total_reward += reward
    
    # Update target network
    if episode % update_target_every == 0:
        agent.update_target_model()
    
    # Print progress
    if episode % print_every == 0:
        avg_reward = total_reward / print_every if episode > 0 else total_reward
        avg_loss = np.mean(losses) if losses else 0
        print(f"Episode: {episode}, Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")
        writer.add_scalar('Avg_Reward', avg_reward, episode)
        writer.add_scalar('Avg_Loss', avg_loss, episode)
        total_reward = 0
        losses = []

# -- Evaluation --
def evaluate_recommendations(model, valid_data, top_k=10):
    # Convert validation data to user-item ratings dictionary
    user_ratings = defaultdict(dict)
    for _, row in valid_data.iterrows():
        user_ratings[row['userId']][row['movieId']] = row['rating']
    
    precision_at_k = []
    recall_at_k = []
    ndcg_at_k = []
    
    model.eval()
    
    for user_id, item_ratings in user_ratings.items():
        # Get actual items the user liked (rated >= 4)
        relevant_items = {item_id for item_id, rating in item_ratings.items() if rating >= 4}
        if not relevant_items:
            continue
            
        # Generate recommendations
        with torch.no_grad():
            user_tensor = torch.tensor([user_id], dtype=torch.long).to(device)
            q_values = model(user_tensor)
            
            # Get top-k items sorted by Q-value
            q_values = q_values.cpu().numpy().flatten()
            already_rated = list(item_ratings.keys())
            for item_id in already_rated:
                q_values[item_id] = -float('inf')  # Exclude already rated items
                
            recommended_items = np.argsort(-q_values)[:top_k]
            
        # Calculate metrics
        hits = len(set(recommended_items) & relevant_items)
        
        # Precision@K: proportion of recommended items that are relevant
        precision = hits / top_k
        precision_at_k.append(precision)
        
        # Recall@K: proportion of relevant items that are recommended
        recall = hits / len(relevant_items)
        recall_at_k.append(recall)
        
        # NDCG@K: discounted cumulative gain normalized
        dcg = 0
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_items), top_k)))
        
        for i, item_id in enumerate(recommended_items):
            if item_id in relevant_items:
                dcg += 1 / np.log2(i + 2)
                
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_at_k.append(ndcg)
    
    return {
        'precision@k': np.mean(precision_at_k),
        'recall@k': np.mean(recall_at_k),
        'ndcg@k': np.mean(ndcg_at_k)
    }

# Evaluate the model
print("\nEvaluating model...")
metrics = evaluate_recommendations(main_model, data_valid, top_k=10)
print(f"Precision@10: {metrics['precision@k']:.4f}")
print(f"Recall@10: {metrics['recall@k']:.4f}")
print(f"NDCG@10: {metrics['ndcg@k']:.4f}")

# Save the model
torch.save(main_model.state_dict(), "rl_recsys_model.pth")
print("Model saved to rl_recsys_model.pth")