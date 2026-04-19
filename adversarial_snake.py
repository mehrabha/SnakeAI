from models.bots import SnakeBot

from snake import SnakeGame
from tkinter import Tk, Canvas


COLORS = [
    '#001A23',  #bg
    '#31493C',  #snake
    '#7A9E7E',  #snake_head
    '#c9d5d6',  #food
    '#493136',  #snake2
    '#9E7A90',  #snake2_head
    '#000d12',  #boundary
]

SIZE = 10
VIEW = 10
PIXEL_SIZE = 40 # Resolution of each box
SPEED = 20

global game
global agent
    
def draw_frame(turn=0):
    canvas.delete("all")

    # Generate a matrix based on game state
    matrix = game.generate_matrix(centered=False)
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
    
    if turn == 0:
        prediction = agent.predict(tuple(game.snake), game.food, tuple(game.snake2))
        
        # Move snake based on prediction
        game.move(prediction, player = 0)
        turn = 1
    else:
        prediction = agent2.predict(tuple(game.snake2), game.food, tuple(game.snake))
        
        # Move snake based on prediction
        game.move(prediction, player = 1)
        turn = 0

    if game.over():
        return
    
    root.after(int(1000 / SPEED), draw_frame, turn)

agent = SnakeBot(SIZE, SIZE)
agent2 = SnakeBot(SIZE, SIZE)
game = SnakeGame(SIZE, SIZE, player2=True)


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