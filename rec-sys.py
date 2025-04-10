import numpy as np
import pandas as pd
from sklearn import model_selection, metrics, preprocessing
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# -- Use GPU if available --
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Read & Analyze Dataset --
data = pd.read_csv("RecSys_torch/ml-latest-small/ratings.csv")
data.info()
print("Number of unique users: ", data.userId.nunique())
print("Number of unique movies: ", data.movieId.nunique())
print("Distribution of ratings: ", data.rating.value_counts())
print("Dataset size: ", data.shape)

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

class RecSysModel(nn.Module):
    def __init__(self, n_users, n_movies):
       super(RecSysModel, self).__init__()

       self.user_embed = nn.Embedding(n_users, 32)
       self.movie_embed = nn.Embedding(n_movies, 32)
       self.out = nn.Linear(64,1)

    def forward(self, users, movies, ratings=None):
        users = users.to(device)
        movies = movies.to(device)
        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        output = torch.cat([user_embeds, movie_embeds], dim=1)
        output = self.out(output)

        return output
    
# Encode user and movie id
label_user = preprocessing.LabelEncoder()
label_movie = preprocessing.LabelEncoder()
data.userId = label_user.fit_transform(data.userId.values)
data.movieId = label_movie.fit_transform(data.movieId.values)

# -- Split train and test dataset
data_train, data_valid = model_selection.train_test_split(
    data, test_size=0.2, random_state=42, stratify=data.rating.values
)

train_dataset = MovieDataset(
    users=data_train.userId.values,
    movies=data_train.movieId.values,
    ratings=data_train.rating.values
)
print("Train dataset size:", len(train_dataset))

valid_dataset = MovieDataset(
    users=data_valid.userId.values, 
    movies=data_valid.movieId.values,
    ratings=data_valid.rating.values
)
print("Test dataset size:", len(valid_dataset))

# -- Initiate data loader, batch size=4 --
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=2)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=2)

dataiter = iter(train_loader)
dataloader_data = next(dataiter)
print(dataloader_data)


# -- Define Model, Optimizer, and Loss Function --
model = RecSysModel(n_users=len(label_user.classes_),n_movies=len(label_movie.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters())
sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)
loss_func = nn.MSELoss()

print(len(label_user.classes_))
print(len(label_movie.classes_))
print(data.movieId.max())
print(len(train_dataset))

with torch.no_grad():  
    model_output = model(dataloader_data['users'],dataloader_data['movies'])
    print(f"Model output: {model_output}, size:, {model_output.size()}")

# -- Training Loop --
epoch = 1
total_loss = 0
plot_steps, print_steps = 5000, 5000
step_cnt = 0
all_losses_list =[]

model.train()
print(f"Training on:, {device}")
for e in range(epoch):
    for i,train_data in enumerate(train_loader):
        output = model(train_data['users'],train_data['movies'])

        rating = train_data['ratings'].view(4,-1).to(torch.float32)
        rating = rating.to(device)
        loss = loss_func(output,rating)
        total_loss = total_loss + loss.sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_cnt = step_cnt + len(train_data['users'])

        #Plot every 5000 steps
        if(step_cnt % plot_steps == 0):
            avg_loss = total_loss/(len(train_data['users']) * plot_steps)
            print(f"epoch {e} loss at step {step_cnt} is {avg_loss}")
            total_loss = 0 # Reset total loss

# -- Plot Loss --
plt.figure()
plt.plot(all_losses_list)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.show()