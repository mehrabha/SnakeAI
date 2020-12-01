from snake import SnakeGame
from models.dq_agent import Agent
from models.neural_nets import NeuralNetwork, NeuralNetworkSingle

global game
global agent

WIDTH, HEIGHT = (6, 6) # Matrix size
PATH = './nn/'
#FILENAME = str(WIDTH) + 'x' + str(HEIGHT) + '.pth'
FILENAME = 's6_512.pth'
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
        return -1 * (game.score() ** 1.5)
    
    score_diff = game.score() - score_old
    if score_diff > 0:
        return game.score() ** 1.4
    
    if game.get_distance() < dist_old:
        if len(game.snake) + 20 > game.steps_since_last:
            return 3 / len(game.snake)
        else:
            return 0
    elif game.get_distance() == dist_old:
        return 0
    else:
        return -3 * (1 + game.steps_since_last / 50) / len(game.snake)
        
            

def train(path, loops, steps, eps=.03, decay=0.5, eps_min=.03, reset=False):
    if not reset:
        agent.load_nn(path)
    else:
        agent.save_nn(path)
        
    for i in range(loops):
        print('Loop:', i, '- Current eps=', round(eps, 5), '- file=', path)
        lengths = 0
        num_games = 0
        games_won = 0
        for j in range(steps):
            state = game.get_flat_matrix()
            prediction = agent.predict(state, eps)
            reward = reward_function(game, prediction)
            new_state = game.get_flat_matrix()
            terminated = game.over()
            
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
game = SnakeGame(WIDTH, HEIGHT)

# Initialize Neural Net
nn = NeuralNetworkSingle(inp_dim=[WIDTH * HEIGHT + 22], out_dim=4, l1_dim=512,
                         lr=.00005)

# Initialize Deep Q Agent
agent = Agent(nn=nn, inp_dim=[WIDTH * HEIGHT + 22], out_dim=4, 
              gamma=.99, batch_size=64, mem_size=20000)

# Run training loop
train(PATH + FILENAME, loops=500, steps=10000, eps=1, 
      decay=.95, eps_min=.0001, reset=True)


