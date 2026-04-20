from models.bots import MinimaxSnakeAgent, GreedySnakeAgent

from game.snake import SnakeGame
from tkinter import Tk, Canvas


COLORS = [
    '#00131A',  #bg
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
SPEED = 30

global game
global agent
    
PLAYER1_WINS = 0
PLAYER2_WINS = 0

def draw_frame(turn=0):
    global PLAYER1_WINS, PLAYER2_WINS
    
    canvas.delete("all")

    # Generate a matrix based on game state
    matrix = game.generate_matrix(centered=False)
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
    
    if turn == 0:
        prediction = agent.predict(game.snake.copy(), game.food, game.snake2.copy())
        
        # Move snake based on prediction
        game.move(prediction, player = 0)
        turn = 1
    else:
        prediction = agent2.predict(game.snake2.copy(), game.food, game.snake.copy())
        
        # Move snake based on prediction
        game.move(prediction, player = 1)
        turn = 0

    if game.over():
        print('Game Over, Player {} wins!'.format(2 if turn == 0 else 1))

        if turn == 0:
            PLAYER1_WINS += 1
        else:
            PLAYER2_WINS += 1

        print("Win counts: PLAYER 1 = {}, PLAYER 2 = {}". format(PLAYER1_WINS, PLAYER2_WINS))

        if (PLAYER1_WINS + PLAYER2_WINS) % 10 == 0:
            print("{} game win counts: PLAYER 1 = {}, PLAYER 2 = {}". format(PLAYER1_WINS + PLAYER2_WINS, PLAYER1_WINS, PLAYER2_WINS))
            proceed = input("Continue(Y/N)?")

            if proceed != 'Y':
                return
        print(".....")
        
        
        game.begin()
        turn = 0
    
    root.after(int(1000 / SPEED), draw_frame, turn)

def get_color(val):
    if isinstance(val, int):
        return COLORS[val]
    elif isinstance(val, str):
        if 'body1' in val:
            hexstr = COLORS[1][1: ]
        elif 'body2' in val:
            hexstr = COLORS[4][1: ]

        rgb = tuple(bytes.fromhex(hexstr))
        color_intensity = .97 ** (int(val.split('-')[1]) - 1)

        rgb = (rgb[0] * color_intensity, rgb[1] * color_intensity, rgb[2] * color_intensity)

        rgb = [int(rgb[0]), int(rgb[1]), int(rgb[2])]

        return '#{:02x}{:02x}{:02x}'.format(*rgb)

agent = MinimaxSnakeAgent(SIZE, SIZE)
agent2 = GreedySnakeAgent(SIZE, SIZE)
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