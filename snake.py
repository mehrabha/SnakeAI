import numpy as np
import random

# [down, right, up, left]
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
        
        vision = np.zeros(12, dtype=np.float32)
        
        # Neighbor blocks
        for i in range(len(move_data)):
            x = move_data[i][0] + head[0]
            y = move_data[i][1] + head[1]
            
            valid = (0 <= x < self.shape[0] and
                     0 <= y < self.shape[1])
            
            if valid:
                if matrix[x][y] == 0:
                    vision[i] = 0
                elif matrix[x][y] == 3:
                    vision[i] = 1
                else:
                    vision[i] = -1
            else:
                vision[i] = -1
        
        # Food location
        vision[4] = food[0]
        vision[5] = food[1]
        
        # Is food reachable
        vision[6] = self.is_reachable(head, self.generate_matrix())
        
        # Head location
        vision[7] = head[0]
        vision[8] = head[1]
        
        # Length
        vision[9] = self.score()
        
        # Steps since last food
        vision[10] = self.steps_since_last
        
        # Direction
        vision[11] = self.dir
        
        '''
        print('Down:', vision[0], 'Right:', vision[1], 'Up:', vision[2], 'Left:', vision[3])
        print('Food Location:', vision[4], vision[5], 'Distance:', vision[6])
        print('Head:', vision[7], vision[8], 'Tail:', vision[9], vision[10])
        print('Direction:', vision[11])
        print('------------------------------------------------\n')
        '''
        return np.concatenate([matrix.flatten(), vision])

    def over(self):
        return not self.status

    def is_reachable(self, start, mtx, visited=None):
        if visited is None:
            visited = np.zeros(shape=self.shape, dtype=np.bool)
        
        x = start[0]
        y = start[1]
        
        xy_in_range = (0 <= x < self.shape[0] and
                       0 <= y < self.shape[1])
        
        if not xy_in_range or visited[x][y]:
            return False
        else:
            visited[x][y] = True
        
        # Current position is food
        if mtx[x][y] == 3:
            return True
        
        # Current position is an obstacle
        if mtx[x][y] not in [0, 2]:
            return False
        
        return (
            # Check left
            self.is_reachable((x - 1, y), mtx, visited) or
            # Check right
            self.is_reachable((x + 1, y), mtx, visited) or
            # Check down
            self.is_reachable((x, y + 1), mtx, visited) or
            # Check up
            self.is_reachable((x, y - 1), mtx, visited)
        )
        
    def spawn_food(self):
        if len(self.snake) == self.shape[0] * self.shape[1]:
            self.begin()
        
        x = random.randint(0, self.shape[0] - 1)
        y = random.randint(0, self.shape[1] - 1)
        if (x, y) in self.snake:
            self.spawn_food()
        else:
            self.food = (x, y)

    def score(self):
        return len(self.snake)
    
    def get_distance(self):
        food = self.food
        snake = self.snake
        x = abs(food[0] - snake[-1][0])
        y = abs(food[1] - snake[-1][1])
        return x + y

