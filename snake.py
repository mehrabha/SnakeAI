import numpy as np
import random

move_data = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class SnakeGame:
    def __init__(self, width, height):
        self.snake = []
        self.food = (0, 0)
        self.shape = (width, height)
        self.status = True


    def begin(self):
        x = int(self.shape[0]/3)
        y = int(self.shape[1]/3)
        self.snake.extend([(x - 1, y), (x, y), (x + 1, y)])
        self.spawn_food()
        self.status = False

    def move(self, direction):
        snake = self.snake
        print(snake)
        for i in range(len(snake) - 1):
            snake[i] = snake[i + 1]

        # head
        x = snake[-1][0] + move_data[direction][0]
        y = snake[-1][1] + move_data[direction][1]
        snake[-1] = (x, y)


    def generate_matrix(self):
        matrix = np.zeros(shape=self.shape, dtype=np.int32)

        for x, y in self.snake:
            matrix[x][y] = 1
        
        head = self.snake[-1]
        matrix[head[0]][head[1]] = 2

        food = self.food
        matrix[food[0]][food[1]] = 3

        return matrix

    def over(self):
        return self.status


    def spawn_food(self):
        x = random.randint(0, self.shape[0] - 1)
        y = random.randint(0, self.shape[1] - 1)

        if (x, y) in self.snake:
            self.spawn_food()
        else:
            self.food = (x, y)

