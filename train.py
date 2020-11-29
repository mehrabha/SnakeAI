from snake import SnakeGame
from models.dq_agent import Agent

global game
global agent

WIDTH, HEIGHT = (6, 6) # Matrix size
PATH = './nn/'
FILENAME = '6x6.pth'

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
        return game.score() ** 1.5
    
    if game.get_distance() < dist_old:
        if len(game.snake) + 20 > game.steps_since_last:
            return 3 / len(game.snake)
        else:
            return 0
    return 3 * (dist_old - game.get_distance()) / len(game.snake)
        
            

def train(path, loops, steps, eps=.03, decay=0, eps_min=.03, reset=False):
    if not reset:
        agent.load_nn(path)
    else:
        agent.save_nn(path)
        
    for i in range(loops):
        print('Loop:', i, '- Current eps=', round(eps, 3), '- file=', path)
        lengths = 0
        num_games = 0
        for j in range(steps):
            state = game.get_flat_matrix()
            prediction = agent.predict(state, eps)
            reward = reward_function(game, prediction)
            new_state = game.get_flat_matrix()
            terminated = game.over()
            
            agent.store(state, prediction, reward, new_state, terminated)
            agent.learn()
            
            if game.over():
                num_games += 1
                lengths += game.score()
                game.begin()
            
        # Save state every few runs
        if i % 10 == 0:
            agent.save_nn(path)
        eps = max(eps * decay, eps_min)
        if num_games > 0:
            print(' - Loop:', i, 'avg score:', round(lengths/num_games, 2))
    agent.save_nn(path)

game = SnakeGame(WIDTH, HEIGHT)


agent = Agent(inp_dim=[WIDTH * HEIGHT + 12], l1_dim=256, l2_dim=256, out_dim=4, 
              gamma=.99, lr=.00003,
              batch_size=64, mem_size=50000)

train(PATH + FILENAME, loops=500, steps=10000, eps=.001, 
      decay=.95, eps_min=.001, reset=False)


