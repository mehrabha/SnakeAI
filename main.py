from snake import SnakeGame
from models.dq_agent import Agent
from models.neural_nets import NeuralNetwork, NeuralNetworkSingle
from tkinter import Tk, Canvas


COLORS = [
    '#001A23',
    '#31493C',
    '#7A9E7E',
    '#c9d5d6'
]

PIXEL_SIZE = 35 # Resolution of each box

SPEED = 10

WIDTH, HEIGHT = (6, 6) # Matrix size
PATH = './nn/'
FILENAME = 's6_256x256.pth'

global game
global agent
    
def draw_frame():
    canvas.delete("all")

    # Generate a matrix based on game state
    matrix = game.generate_matrix()
    state = game.get_flat_matrix()
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
    
    # Deep Learning Agent
    prediction = agent.predict(state)
    # Move snake based on prediction
    game.move(prediction)
    # Restart on collision
    if game.over() or game.won():
        print('Score:', game.score(), 'Steps:', game.steps)
        game.begin()
        
    
    root.after(int(1000 / SPEED), draw_frame)


#Initialize nn
nn = NeuralNetwork(inp_dim=[WIDTH * HEIGHT + 22], out_dim=4, 
                   l1_dim=256, l2_dim=256)

# Initialize deep learning agent
agent = Agent(nn=nn, inp_dim=[WIDTH * HEIGHT + 22], out_dim=4)

# Load nn
agent.load_nn(PATH + FILENAME)

game = SnakeGame(WIDTH, HEIGHT)
agent.predict(game.get_flat_matrix())

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
