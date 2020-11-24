from snake import SnakeGame
from models.dq_agent import Agent

global game
global agent

WIDTH, HEIGHT = (10, 10) # Matrix size
PATH = './nn/'
FILENAME = 'nn.pth'

def train(nsteps, randomness=0):
    for i in range(int(nsteps)):
        state = game.get_float_matrix()
        
        # Before move
        score_diff = game.score()
        distance_rewards = game.get_distance()
        
        # Move snake based on prediction
        prediction = agent.predict(state, randomness)
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


training_steps = 100000
agent = Agent(inp_dim=[WIDTH * HEIGHT], out_dim=4, gamma=0, lr=.03,
              eps=0, eps_min=0,eps_decay=0,
              batch_size=256, mem_size=50000) # Initialize agent
#agent = SnakeBot(WIDTH, HEIGHT)

game = SnakeGame(WIDTH, HEIGHT)
agent.save_nn(PATH + FILENAME)
ep = .8
for i in range(95):
    print('Group', i, of 95,  '  EP:', ep)
    train(training_steps, randomness=ep)
    ep -= .08
    agent.save_nn(PATH + FILENAME)
    