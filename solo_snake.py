from models.bots import GreedySnakeAgent

from game.snake import SnakeGame
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

SIZE = 9
VIEW = 9
PIXEL_SIZE = 40 # Resolution of each box
SPEED = 10

global game
global agent
    
def draw_frame():
    canvas.delete("all")

    # Generate a matrix based on game state
    matrix = game.generate_matrix()

    for i in range(VIEW):
        for j in range(VIEW):
            color = get_color(matrix[i][j])
            border = int(PIXEL_SIZE * .04)
            x = i * PIXEL_SIZE
            y = j * PIXEL_SIZE

            canvas.create_rectangle(
                x + border, y + border,
                x + PIXEL_SIZE - border,
                y + PIXEL_SIZE - border,
                fill=color,
                outline=COLORS[0]
            )
    
    prediction = agent.predict(game.snake, game.food)
    
    
    # Move snake based on prediction
    game.move(prediction)
    # Restart on collision
    if game.over():
        game.begin()
    
    root.after(int(1000 / SPEED), draw_frame)

def get_color(val):
    if isinstance(val, int):
        return COLORS[val]
    elif isinstance(val, str):
        if 'body1' in val:
            hexstr = COLORS[1][1: ]
        elif 'body2' in val:
            hexstr = COLORS[4][1: ]

        rgb = tuple(bytes.fromhex(hexstr))
        color_intensity = .98 ** (int(val.split('-')[1]) - 1)

        rgb = (rgb[0] * color_intensity, rgb[1] * color_intensity, rgb[2] * color_intensity)

        rgb = [int(rgb[0]), int(rgb[1]), int(rgb[2])]

        return '#{:02x}{:02x}{:02x}'.format(*rgb)

agent = GreedySnakeAgent(SIZE, SIZE)
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