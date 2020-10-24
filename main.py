import numpy as np
from snake import SnakeGame
from models.bots import SnakeBot
from tkinter import Tk, Canvas

COLORS = [
    '#001A23',
    '#31493C',
    '#7A9E7E',
    '#c9d5d6'
]

WIDTH, HEIGHT = (16, 9)
PIXEL_SIZE = 35
SPEED = 8

global game
global agent

def draw_frame():
    canvas.delete("all")
    matrix = game.generate_matrix()
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
    prediction = agent.predict(game.snake, game.food)
    game.move(prediction)
    
    if game.over():
        game.begin()
    root.after(int(1000 / SPEED), draw_frame)


# game
resolution_x = PIXEL_SIZE * WIDTH
resolution_y = PIXEL_SIZE * HEIGHT
game = SnakeGame(WIDTH, HEIGHT)
agent = SnakeBot(WIDTH, HEIGHT)
game.begin()

root = Tk()
root.title('Snake AI')
canvas = Canvas(root, bg=COLORS[0], width=resolution_x, height=resolution_y)
canvas.pack()

root.after(100, draw_frame)
root.mainloop()
