from models.bots import SnakeBot

from snake import SnakeGame
from tkinter import Tk, Canvas


COLORS = [
    '#001A23',
    '#31493C',
    '#7A9E7E',
    '#c9d5d6',
    '#000d12'
]

SIZE = 9
VIEW = 7
PIXEL_SIZE = 40 # Resolution of each box
SPEED = 2

global game
global agent
    
def draw_frame():
    canvas.delete("all")

    # Generate a matrix based on game state
    matrix = game.generate_matrix(centered=True, view_dist=VIEW)
    print()
    for i in range(VIEW):
        for j in range(VIEW):
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
    
    prediction = agent.predict(game.snake, game.food)
    
    
    # Move snake based on prediction
    game.move(prediction)
    # Restart on collision
    if game.over():
        game.begin()
    
    root.after(int(1000 / SPEED), draw_frame)

agent = SnakeBot(SIZE, SIZE)
game = SnakeGame(SIZE, SIZE)


# game
resolution_x = PIXEL_SIZE * VIEW
resolution_y = PIXEL_SIZE * VIEW

game.begin()
root = Tk()
root.title('Snake AI')
canvas = Canvas(root, bg=COLORS[0], width=resolution_x, height=resolution_y)
canvas.pack()

root.after(100, draw_frame) # Visualize agent playing the game
root.mainloop()