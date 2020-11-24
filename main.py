import numpy as np
import torch as t
from snake import SnakeGame
from models.bots import SnakeBot
from models.dq_agent import Agent, NeuralNetwork
from tkinter import Tk, Canvas

COLORS = [
    '#001A23',
    '#31493C',
    '#7A9E7E',
    '#c9d5d6'
]

WIDTH, HEIGHT = (10, 10) # Matrix size
PIXEL_SIZE = 35 # Resolution of each box
SPEED = 10

global game
global agent


def train(nsteps, randomness=0):
    for i in range(int(nsteps)):
        if i % 1000 == 0:
            print("Progress:", i, 'trained out of', nsteps)
            
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
            score_diff = -5
            distance_rewards = -1
        else:
            score_diff = (game.score() - score_diff) * 5
            if score_diff > 0:
                distance_rewards = 0
            else:
                distance_rewards -= game.get_distance()
            
        reward = 5*score_diff + distance_rewards
        agent.store(state, prediction, reward)
        agent.learn()
    
def draw_frame():
    canvas.delete("all")

    # Generate a matrix based on game state
    matrix = game.generate_matrix()
    state = game.get_float_matrix()
    for i in range(WIDTH):
        for j in range(HEIGHT):
            color = matrix[i][j]
            border = int(PIXEL_SIZE * .05)
            x = i * PIXEL_SIZE
            y = j * PIXEL_SIZE

            canvas.create_rectangle(
                x + border, y + border,
                x + PIXEL_SIZE - border,
                y + PIXEL_SIZE - border,
                fill=COLORS[color],
                outline=COLORS[0]
            )
    
    # The AI controller
    #prediction = agent.predict(game.snake, game.food)
    
    # Deep Learning Agent
    prediction = agent.predict(state)
    
    # Move snake based on prediction
    game.move(prediction)
    # Restart on collision
    if game.over():
        game.begin()
    
    root.after(int(1000 / SPEED), draw_frame)


# game
resolution_x = PIXEL_SIZE * WIDTH
resolution_y = PIXEL_SIZE * HEIGHT
training_steps = 10000
agent = Agent(inp_dim=[WIDTH * HEIGHT], out_dim=4, gamma=0, lr=.03,
              eps=0, eps_min=0,eps_decay=0,
              batch_size=0, mem_size=20000) # Initialize agent
#agent = SnakeBot(WIDTH, HEIGHT)

game = SnakeGame(WIDTH, HEIGHT)
ep = .64
for i in range(5):
    print('Group', i, '( ep =', ep, '):')
    train(training_steps, randomness=ep)
    ep *= .5

game.begin()
root = Tk()
root.title('Snake AI')
canvas = Canvas(root, bg=COLORS[0], width=resolution_x, height=resolution_y)
canvas.pack()

root.after(100, draw_frame) # Visualize agent playing the game
root.mainloop()
