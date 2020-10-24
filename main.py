import numpy as np
from snake import SnakeGame
from tkinter import Tk, Canvas

COLORS = [
    '#001A23',
    '#31493C',
    '#7A9E7E',
    '#c9d5d6'
]

WIDTH, HEIGHT = (16, 9)
PIXEL_SIZE = 35
GAME_SPEED = 5

def draw_frame():
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
    root.after(400, draw_frame)


# game
resolution_x = PIXEL_SIZE * WIDTH
resolution_y = PIXEL_SIZE * HEIGHT

root = Tk()
root.title('Snake AI')
canvas = Canvas(root, bg=COLORS[0], width=resolution_x, height=resolution_y)
canvas.pack()

root.after(400, draw_frame)
root.mainloop()
