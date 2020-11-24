# [down, right, up, left]
move_data = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Returns manhatan distance
def distance(x, y):
    return abs(y[0] - x[0]) + abs(y[1] - x[1])


class SnakeBot:
    """
    Class representing an FSM based agent


    Methods
    -------
    predict(list, tuple):
        Given a list containing snake's location, choose a direction for snake

    
    """


    def __init__(self, width, height):
        self.shape = (width, height)


    def predict(self, snake, food):
        """

        Params
        ------
        snake : list of tuples
            List containing the locations of snake.
            Last element in the list is head, rest are tails
        food : tuple
            Location of food.

        Returns
        -------
        int :
            Direction of snake
        """

        dists = []

        # Iterate through the move_data array [up, right, down, left],
        # find a direction that results in the shortest distance to food from head
        for idx in range(4):
            # head + direction = new head x
            new_dir = (snake[-1][0] + move_data[idx][0], snake[-1][1] + move_data[idx][1])

            # check collision
            valid = (
                0 <= new_dir[0] < self.shape[0] and
                0 <= new_dir[1] < self.shape[1] and
                new_dir not in snake
            )
            
            if valid:
                num_traps = self.check_traps(snake, new_dir) ** self.get_state(snake)
                min_dist = distance(new_dir, food)
                dists.append((num_traps + min_dist, idx))
        
        dists.sort()
        if len(dists) > 0:
            return dists[0][1]
        else:
            return 0

    def check_traps(self, snake, dir):
        result = 0

        for x, y in move_data:
            new_dir = (dir[0] + x, dir[1] + y)
            blocked = (
                not (0 <= new_dir[0] < self.shape[0]) or
                not (0 <= new_dir[1] < self.shape[1]) or
                new_dir in snake
            )
            
            if blocked and new_dir != snake[-1]:
                result += 1
        return result

    def get_state(self, snake):
        if len(snake) < 12:
            return 1
        elif len(snake) < 24:
            return 2
        else:
            return 3
