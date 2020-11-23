import numpy as np
import torch as t
from snake import SnakeGame
from models.bots import SnakeBot
from models.dq_agent import Agent, NewralNetwork
from tkinter import Tk, Canvas

import threading
import time

COLORS = [
    '#001A23',
    '#31493C',
    '#7A9E7E',
    '#c9d5d6'
]

WIDTH, HEIGHT = (8, 8) # Matrix size
PIXEL_SIZE = 35 # Resolution of each box
SPEED = 10

global game
global agent


def train(nsteps, randomness=0):
    for i in range(int(nsteps)):
        if i % 1000 == 0:
            print("Progress:", i, 'trained out of', nsteps)
            
        state = game.get_tensor()
        
        # The AI controller
        prediction = agent.predict(state, randomness)
        
        # Move snake based on prediction
        score_diff = game.score()
        distance_rewards = game.get_distance()
        game.move(prediction)
        
        # Restart on collision, adjust score
        if game.over():
            game.begin()
            score_diff = -5
            distance_rewards = -1
        else:
            score_diff = (game.score() - score_diff) * 5
            distance_rewards -= game.get_distance()
            
        # Train AI for every step
        agent.learn()
        reward = 5*score_diff + distance_rewards
        agent.store(state, prediction, reward)
    
def draw_frame():
    canvas.delete("all")

    # Generate a matrix based on game state
    matrix = game.generate_matrix()
    state = game.get_tensor()
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
training_steps = 500000
agent = Agent(inp_dim=[WIDTH * HEIGHT], out_dim=4, 
              mem_size=256) # Initialize agent
#agent = SnakeBot(WIDTH, HEIGHT)

game = SnakeGame(WIDTH, HEIGHT)

train(training_steps)

game.begin()
root = Tk()
root.title('Snake AI')
canvas = Canvas(root, bg=COLORS[0], width=resolution_x, height=resolution_y)
canvas.pack()

root.after(100, draw_frame) # Visualize agent playing the game
root.mainloop()
