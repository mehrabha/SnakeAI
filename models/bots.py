move_data = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def distance(x, y):
    return abs(y[0] - x[0]) + abs(y[1] - x[1])

class SnakeBot:
    def __init__(self, width, height):
        self.shape = (width, height)

    def predict(self, snake, food):
        prediction = 0
        min_dist = 2**30

        for i in range(4):
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