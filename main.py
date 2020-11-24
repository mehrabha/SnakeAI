from snake import SnakeGame
from models.bots import SnakeBot
from models.dq_agent import Agent
from tkinter import Tk, Canvas


COLORS = [
    '#001A23',
    '#31493C',
    '#7A9E7E',
    '#c9d5d6'
]

WIDTH, HEIGHT = (10, 10) # Matrix size
PIXEL_SIZE = 35 # Resolution of each box
SPEED = 8

PATH = './nn/'
FILENAME = 'nn.pth'

global game
global agent
    
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


# Initialize deep learning agent
agent = Agent(inp_dim=[WIDTH * HEIGHT], out_dim=4, gamma=0, lr=.03,
              eps=0, eps_min=0,eps_decay=0,
              batch_size=256, mem_size=50000)

# Load nn
agent.load_nn(PATH + FILENAME)

game = SnakeGame(WIDTH, HEIGHT)

# game
resolution_x = PIXEL_SIZE * WIDTH
resolution_y = PIXEL_SIZE * HEIGHT

game.begin()
root = Tk()
root.title('Snake AI')
canvas = Canvas(root, bg=COLORS[0], width=resolution_x, height=resolution_y)
canvas.pack()

root.after(100, draw_frame) # Visualize agent playing the game
root.mainloop()
