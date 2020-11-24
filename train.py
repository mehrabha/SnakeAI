import time
import _thread
from snake import SnakeGame
from models.dq_agent import Agent

global game
global agent

WIDTH, HEIGHT = (10, 10) # Matrix size
PATH = './nn/'
FILENAME = 'nn.pth'

global game
global agent

def train(path, loops, steps, eps=.01, decay=0, eps_min=.01):
    agent.save_nn(path)
    for i in range(loops):
        print('Progress:', round(100 * i/loops, 2),
              '%   Current eps=', round(eps, 3))
        for j in range(steps):
            state = game.get_flat_matrix()
            
            # Before move
            score_diff = game.score()
            distance_rewards = game.get_distance()
            
            # Move snake based on prediction
            prediction = agent.predict(state, eps)
            game.move(prediction)
            new_state = game.get_flat_matrix()
            terminated = game.over()
            
            # Restart on collision, adjust score
            if game.over():
                game.begin()
                score_diff = -1
                distance_rewards = -1
            else:
                score_diff = (game.score() - score_diff)
                if score_diff > 0:
                    distance_rewards = 0
                else:
                    distance_rewards -= game.get_distance()
            
            reward = score_diff + distance_rewards * .1
            agent.store(state, prediction, reward, new_state, terminated)
            agent.learn()
            
        # Save state every few runs
        if i % 10 == 0:
            agent.save_nn(path)
        eps = max(eps - decay, eps_min)

game = SnakeGame(WIDTH, HEIGHT)
agent = Agent(inp_dim=[WIDTH * HEIGHT], out_dim=4, gamma=.99, lr=.03,
          batch_size=256, mem_size=100000)
train(PATH + FILENAME, loops=10, steps=1000)
    
    