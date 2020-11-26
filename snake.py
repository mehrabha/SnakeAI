import numpy as np
import torch as t
import random

# [up, right, down, left]
move_data = [(0, 1), (1, 0), (0, -1), (-1, 0)]

class SnakeGame:
    """
    Class to represent a game of snake


    Attributes
    ----------
    snake : list of tuples
        List containing the locations of snake.
        Last element in the list is head, rest are tails
    food : tuple
        Location of food.
    shape : tuple
        Matrix size
    status : bool
        True while game is running


    Methods
    -------
    begin():
        Restart game, starting snake length = 3
    move(int):
        Move snake using the move_data array, check collision and food.
    generate_matrix():
        Return a width by height matrix representing the current frame.
    over():
        Returns true if game over
    """

    
    def __init__(self, width, height):
        self.snake = []
        self.food = (0, 0)
        self.shape = (width, height)
        self.steps_since_last = 0
        self.status = False
        self.dir = 1
        self.begin()


    def begin(self):
        x = int(self.shape[0]/2)
        y = int(self.shape[1]/2)
        self.snake = [(x - 1, y), (x, y), (x + 1, y)]
        self.spawn_food()
        self.steps_since_last = 0
        self.status = True
        self.dir = 1


    def move(self, direction):
        if not self.status:
            return
        
        backwards = (
            (self.dir == 0 and direction == 2) or
            (self.dir == 2 and direction == 0) or
            (self.dir == 1 and direction == 3) or
            (self.dir == 3 and direction == 1)
        )

        if not backwards and self.dir != direction:
            self.dir = direction

        snake = self.snake
        # head
        x = snake[-1][0] + move_data[self.dir][0]
        y = snake[-1][1] + move_data[self.dir][1]

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
                self.steps_since_last = 0
            else:
            # move
                for i in range(len(snake) - 1):
                    snake[i] = snake[i + 1]
                snake[-1] = (x, y)
                self.steps_since_last += 1


    def generate_matrix(self):
        matrix = np.zeros(shape=self.shape, dtype=np.int32)

        for x, y in self.snake:
            matrix[x][y] = 1
            
        head = self.snake[-1]
        matrix[head[0]][head[1]] = 2

        food = self.food
        matrix[food[0]][food[1]] = 3

        return matrix
    
    def get_flat_matrix(self):
        matrix = np.zeros(self.shape, dtype=np.float32)

        for i in range(len(self.snake)):
            x = self.snake[i][0]
            y = self.snake[i][1]
            matrix[x][y] = i + 4

        head = self.snake[-1]
        matrix[head[0]][head[1]] = 2

        food = self.food
        matrix[food[0]][food[1]] = 3
        
        return matrix.flatten()

    def over(self):
        return not self.status


    def spawn_food(self):
        x = random.randint(0, self.shape[0] - 1)
        y = random.randint(0, self.shape[1] - 1)
        if (x, y) in self.snake:
            self.spawn_food()
        else:
            self.food = (x, y)

    def score(self):
        return len(self.snake)**2
    
    def get_distance(self):
        food = self.food
        snake = self.snake
        x = abs(food[0] - snake[-1][0])
        y = abs(food[1] - snake[-1][1])
        return x + y

