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

def train(path, steps, eps=.8, decay=.002):
    agent.save_nn(path)
    for i in range(steps):
        print('Progress:', round(100 * i/steps, 2),
              '%   Current eps=', round(eps, 3))
        for j in range(100000):
            state = game.get_float_matrix()
            
            # Before move
            score_diff = game.score()
            distance_rewards = game.get_distance()
            
            # Move snake based on prediction
            prediction = agent.predict(state, eps)
            game.move(prediction)
            
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
                
            reward = 5*score_diff + distance_rewards * .5
            agent.store(state, prediction, reward)
            agent.learn()
            
        # Save state every few runs
        if i % 10 == 0:
            agent.save_nn(path)
        eps -= decay

game = SnakeGame(WIDTH, HEIGHT)
agent = Agent(inp_dim=[WIDTH * HEIGHT], out_dim=4, gamma=0, lr=.03,
          batch_size=256, mem_size=100000)
train(PATH + FILENAME, steps=390)
    
    