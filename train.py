import numpy as np
from snake import SnakeGame
from models.dq_agent import Agent
from models.neural_nets import NeuralNetwork
from models.replay_buffer import ReplayBuffer

global game
global agent

SIZE = 13
VIEW = 7
PRIO = 1
PATH = './nn/'
FILENAME = 'model1.pth'

def reward_function(game, prediction):
    # Before move
    score_old = game.score()
    dist_old = game.get_distance()
    
    # Move snake based on prediction
    game.move(prediction)
    
    # Restart on collision, adjust score
    if game.over():
        return -1 * (game.score() ** 1.25)
    
    score_diff = game.score() - score_old
    if score_diff > 0:
        return game.score() ** 1.2
    
    if game.get_distance() < dist_old:
        if len(game.snake) + 40 > game.steps_since_last:
            return 3 / len(game.snake)
        else:
            return 0
    elif game.get_distance() == dist_old:
        return 0
    else:
        return -3 * (1 + game.steps_since_last / 100) / len(game.snake)
        
            

def train(game, agent, path, loops, steps, eps=.03, decay=0.5, 
          eps_min=.03, new=True):
    if not new:
        agent.load_nn(path)
    else:
        agent.save_nn(path)
        
    for i in range(loops):
        print('Loop:', i, '- Current eps=', round(eps, 5), '- file=', path)
        lengths = 0
        max_length = 0
        num_games = 0
        games_won = 0
        for j in range(steps):
            state = game.generate_matrix(centered=True, view_dist=VIEW, 
                                         flatten=True, r_type=np.float32)
            prediction = agent.predict(state, eps)
            reward = reward_function(game, prediction)
            new_state = game.generate_matrix(centered=True, view_dist=VIEW, 
                                             flatten=True, r_type=np.float32)
            terminated = game.over()
            
            # Prioritize bigger snakes
            choice = False
            if game.score() > max_length:
                max_length = game.score()
                choice = True
            else:
                random = np.random.random()
                if random < game.score() / max_length:
                    choice = True
            
            if choice:
                agent.store(state, prediction, reward, new_state, terminated)
            agent.learn()
            
            if game.over() or game.won():
                if game.won():
                    games_won += 1
                num_games += 1
                lengths += game.score()
                game.begin()
            
        # Save state every few runs
        if i % 5 == 0:
            agent.save_nn(path)
        eps = max(eps * decay, eps_min)
        if num_games > 0:
            print(' - games:', num_games, ', avg score:', 
                  round(lengths/num_games, 2), ', games won:', games_won)
    agent.save_nn(path)


# Initialize game
game = SnakeGame(SIZE, SIZE)

# Initialize Neural Net
nn = NeuralNetwork(inp_dim=[VIEW * VIEW + 8], out_dim=4, 
                   l1_dim=256, l2_dim=128, lr=.0001)

# Initialize memory
memory = ReplayBuffer(inp_dim=[VIEW * VIEW + 8], mem_size=100000, 
                      batch_size=64, priority_scale=PRIO)

# Initialize Deep Q Agent
agent = Agent(nn=nn, inp_dim=[VIEW * VIEW + 8], out_dim=4, 
              memory=memory, gamma=.99)

# Run training loop
train(game, agent, PATH + FILENAME, loops=100, steps=1000, eps=0, 
      decay=.90, eps_min=.001, new=True)


