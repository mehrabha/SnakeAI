import numpy as np
import random

move_data = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class SnakeGame:
    def __init__(self, width, height):
        self.snake = []
        self.food = (0, 0)
        self.shape = (width, height)
        self.status = False


    def begin(self):
        x = int(self.shape[0]/random.choice([2,3,4]))
        y = random.choice(range(self.shape[1]))
        self.snake = [(x - 1, y), (x, y), (x + 1, y)]
        self.spawn_food()
        self.status = True

    def move(self, direction):
        if not self.status:
            return

        snake = self.snake
        # head
        x = snake[-1][0] + move_data[direction][0]
        y = snake[-1][1] + move_data[direction][1]

        # check collision
        self.status = (
            0 <= x < self.shape[0] and
            0 <= y < self.shape[1] and
            (x, y) not in snake
        )
        
        if self.status:
            # check food
            food = self.food
            if x == food[0] and y == food[1]:
                snake.append(food)
                self.spawn_food()
            else:
            # move
                for i in range(len(snake) - 1):
                    snake[i] = snake[i + 1]
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
        return not self.status


    def spawn_food(self):
        x = random.randint(0, self.shape[0] - 1)
        y = random.randint(0, self.shape[1] - 1)

        if (x, y) in self.snake:
            self.spawn_food()
        else:
            self.food = (x, y)

