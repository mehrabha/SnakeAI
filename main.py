import numpy as np
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

WIDTH, HEIGHT = (20, 20) # Matrix size
PIXEL_SIZE = 35 # Resolution of each box
SPEED = 10

global game
global agent


def train(nsteps):
    for i in range(nsteps):
        if i % 1000 == 0:
            print("Progress:", i, 'trained out of', nsteps)
        state = np.asarray(game.generate_matrix(), dtype=np.float32).flatten()
        
        # The AI controller
        prediction = agent.predict(state)
        # Move snake based on prediction
        score_old = game.score()
        game.move(prediction)
        
        # Restart on collision, adjust score
        if game.over():
            game.begin()
            score_new = -1
        else:
            score_new = game.score()
        
        # Train AI for every step
        score_diff = (score_new - score_old)
        agent.store(state, prediction, score_diff)
        agent.learn()
    
def draw_frame():
    canvas.delete("all")

    # Generate a matrix based on game state
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
    state = np.asarray(matrix, dtype=np.float32).flatten()
    
    # The AI controller
    #prediction = agent.predict(game.snake, game.food)
    
    # Deep Learning Agent
    prediction = agent.predict(state)
    
    # Move snake based on prediction
    game.move(prediction)
    print("Score:", game.score()) #, "State:", agent.get_state(game.snake))


    # Restart on collision
    if game.over():
        game.begin()
    
    root.after(int(1000 / SPEED), draw_frame)


# game
resolution_x = PIXEL_SIZE * WIDTH
resolution_y = PIXEL_SIZE * HEIGHT
training_steps = 2000

game = SnakeGame(WIDTH, HEIGHT) # Initialize new game
agent = Agent(inp_dim=[WIDTH * HEIGHT], out_dim=4, 
              mem_size=training_steps) # Initialize agent
#agent = SnakeBot(WIDTH, HEIGHT)

game.begin()
train(training_steps) # Main training loop

root = Tk()
root.title('Snake AI')
canvas = Canvas(root, bg=COLORS[0], width=resolution_x, height=resolution_y)
canvas.pack()

game.begin()
root.after(100, draw_frame) # Visualize agent playing the game
root.mainloop()
