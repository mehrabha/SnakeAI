from snake import SnakeGame
from models.dq_agent import Agent
from models.neural_nets import NeuralNetwork

import numpy as np
from tkinter import Tk, Canvas


COLORS = [
    '#001A23',
    '#31493C',
    '#7A9E7E',
    '#c9d5d6',
    '#000d12'
]

SIZE = 13
VIEW = 7
PIXEL_SIZE = 40 # Resolution of each box
SPEED = 20
PATH = './nn/'
FILENAME = '13,7_static.pth'

global game
global agent
    
def draw_frame():
    canvas.delete("all")

    # Generate a matrix based on game state
    matrix = game.generate_matrix()
    state = game.generate_matrix(centered=True, view_dist=VIEW, 
                                 flatten=True, r_type=np.float32)
    for i in range(SIZE):
        for j in range(SIZE):
            color = matrix[i][j]
            border = int(PIXEL_SIZE * .04)
            x = i * PIXEL_SIZE
            y = j * PIXEL_SIZE

            canvas.create_rectangle(
                x + border, y + border,
                x + PIXEL_SIZE - border,
                y + PIXEL_SIZE - border,
                fill=COLORS[color],
                outline=COLORS[0]
            )
    
    prediction = agent.predict(state)
    
    
    # Move snake based on prediction
    game.move(prediction)
    # Restart on collision
    if game.over():
        game.begin()
    
    root.after(int(1000 / SPEED), draw_frame)


#Initialize nn
nn = NeuralNetwork(inp_dim=[VIEW * VIEW + 8], out_dim=4, 
                   l1_dim=256, l2_dim=128)

# Initialize deep learning agent
agent = Agent(nn=nn, inp_dim=[VIEW * VIEW + 8], out_dim=4)

# Load nn
agent.load_nn(PATH + FILENAME)

game = SnakeGame(SIZE, SIZE)

# game
resolution_x = PIXEL_SIZE * SIZE
resolution_y = PIXEL_SIZE * SIZE

game.begin()
root = Tk()
root.title('Snake AI')
canvas = Canvas(root, bg=COLORS[0], width=resolution_x, height=resolution_y)
canvas.pack()

root.after(100, draw_frame) # Visualize agent playing the game
root.mainloop()
