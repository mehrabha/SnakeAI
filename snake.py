import numpy as np
import random

move_data = [(0, 1), (1, 0), (0, -1), (-1, 0)]
move_names = ['down', 'right', 'up', 'left']

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
        self.steps = 0
        self.steps_since_last = 0
        self.status = False
        self.won_status = False
        self.dir = 1
        self.begin()


    def begin(self):
        x = int(self.shape[0]/2)
        y = int(self.shape[1]/2)
        self.snake = [(x - 1, y), (x, y), (x + 1, y)]
        self.spawn_food()
        self.steps = 0
        self.steps_since_last = 0
        self.status = True
        self.won_status = False
        self.dir = 1


    def move(self, direction):
        if not self.status or self.won_status:
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
                if len(self.snake) < self.shape[0] * self.shape[1]:
                    self.spawn_food()
                else:
                    self.won_status = True
                self.steps_since_last = 0
            else:
            # move
                for i in range(len(snake) - 1):
                    snake[i] = snake[i + 1]
                snake[-1] = (x, y)
                self.steps_since_last += 1
            self.steps += 1
        
        
    def generate_matrix(self, centered=False, view_dist=None, 
                        flatten=False, r_type=np.int32):
        
        view_dist = self.shape[0] if view_dist is None else view_dist
        matrix = np.zeros(shape=(view_dist, view_dist), dtype=r_type)
        
        translation = (0, 0)
        if centered:
            translation = (int(self.shape[0]/ 2) - self.snake[-1][0],
                           int(self.shape[1]/ 2) - self.snake[-1][1])
            if view_dist < self.shape[0]:
                shape = self.shape[0]
                translation = (translation[0] - int((shape-view_dist) / 2), 
                               translation[1] - int((shape-view_dist) / 2))
        for x, y in self.snake:
            new_x = x + translation[0]
            new_y = y + translation[1]
            
            if self.valid_point(new_x, new_y, view_dist):
                matrix[new_x][new_y] = 1

        head = self.snake[-1]
        matrix[head[0] + translation[0]][head[1] + translation[1]] = 2

        food = self.food
        food_x = food[0] + translation[0]
        food_y = food[1] + translation[1]
        
        if self.valid_point(food_x, food_y, view_dist):
            matrix[food_x][food_y] = 3
            
        for i in range(view_dist):
            for j in range(view_dist):
                new_x = i - translation[0]
                new_y = j - translation[1]
                
                if not self.valid_point(new_x, new_y):
                    matrix[i][j] = -1
        
        if not flatten:
            return matrix
        
        vision = np.zeros(8, dtype=np.float32)
        # food dir horizontal
        vision[0] = -1 if food[0] < head[0] else 1 if food[0] > head[0] else 0
        # food dir vertical
        vision[1] = -1 if food[1] < head[1] else 1 if food[1] > head[1] else 0
        
        # Is food reachable
        vision[2] = self.is_reachable(head, self.generate_matrix())
        
        # Distance from food
        vision[3] = self.get_distance()
        
        # Length
        vision[4] = len(self.snake)
        
        # Steps since last food
        vision[5] = self.steps_since_last
        
        # Snake direction
        vision[6] = self.dir
        
        # Tail direction
        vision[7] = self.get_tail_dir()
        
        return np.concatenate([matrix.flatten(), vision])

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
        x = random.randint(0, self.shape[0] - 1)
        y = random.randint(0, self.shape[1] - 1)
        if (x, y) in self.snake:
            self.spawn_food()
        else:
            self.food = (x, y)

    def score(self):
        if self.won_status:
            return 2 * len(self.snake)
        return len(self.snake)
    
    def get_distance(self):
        food = self.food
        snake = self.snake
        x = abs(food[0] - snake[-1][0])
        y = abs(food[1] - snake[-1][1])
        return x + y
    
    def get_tail_dir(self):
        tail = self.snake[0]
        for i in range(len(move_data)):
            new_tail = (tail[0] + move_data[i][0], tail[1] + move_data[i][1])
            if new_tail == self.snake[1]:
                return i

    def valid_point(self, x, y, test=None):
        if test is None:
            return (0 <= x < self.shape[0] and
                    0 <= y < self.shape[1])
        else:
            return (0 <= x < test and
                    0 <= y < test)

    def over(self):
        return not self.status
    
    def won(self):
        return self.won_status

