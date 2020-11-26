import time
import _thread
from snake import SnakeGame
from models.dq_agent import Agent

global game
global agent

WIDTH, HEIGHT = (10, 10) # Matrix size
PATH = './nn/'
FILENAME = 'nn1.pth'

global game
global agent

def reward_function(game, prediction):
    # Before move
    score_old = game.score()
    dist_old = game.get_distance()
    
    # Move snake based on prediction
    game.move(prediction)
    
    # Restart on collision, adjust score
    if game.over():
        return -10
    
    score_diff = game.score() - score_old
    if score_diff > 0:
        return game.score()
    
    if game.get_distance() < dist_old:
        if len(game.snake) + 10 > game.steps_since_last:
            return 10 / game.steps_since_last
    return -1
        
            

def train(path, loops, steps, eps=.01, decay=0, eps_min=.03, reset=False):
    if not reset:
        agent.load_nn(path)
    else:
        agent.save_nn(path)
        
    for i in range(loops):
        print('Loop:', i, ' - Current eps=', round(eps, 3))
        for j in range(steps):
            if steps > 10000 and j % 1000 == 0:
                print(' -', j, 'steps')
                
            state = game.get_flat_matrix()
            prediction = agent.predict(state, eps)
            reward = reward_function(game, prediction)
            new_state = game.get_flat_matrix()
            terminated = game.over()
            
            agent.store(state, prediction, reward, new_state, terminated)
            agent.learn()
            
            if game.over():
                game.begin()
            
        # Save state every few runs
        if i % 10 == 0:
            agent.save_nn(path)
        eps = max(eps - decay, eps_min)
    agent.save_nn(path)

game = SnakeGame(WIDTH, HEIGHT)
agent = Agent(inp_dim=[WIDTH * HEIGHT], out_dim=4, gamma=.4, lr=.03,
          batch_size=64, mem_size=20000)
train(PATH + FILENAME, loops=20, steps=2000, eps=1, decay=.4, reset=True)
    