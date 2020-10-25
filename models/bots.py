# [up, right, down, left]
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

        prediction = 0
        min_dist = 2**30

        # Iterate through the move_data array [up, right, down, left],
        # find a direction that results in the shortest distance to food from head
        for i in range(4):
            # head + direction = new head x
            x = (snake[-1][0] + move_data[i][0], snake[-1][1] + move_data[i][1])
            y = food

            # check collision
            valid = (
                0 <= x[0] < self.shape[0] and
                0 <= x[1] < self.shape[1] and
                x not in snake and
                distance(x, y) < min_dist
            )
            
            if valid:
                prediction = i
                min_dist = distance(x, y)
        return prediction