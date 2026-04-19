from collections import deque

# [down, right, up, left]
move_data = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Returns manhatan distance
def distance(x, y):
    return abs(y[0] - x[0]) + abs(y[1] - x[1])


class GreedySnakeAgent:
    """
    Class representing an FSM based agent


    Methods
    -------
    predict:
        Given a list containing snake's location, choose a direction for snake

    collsiion:
        Helper function to check for valid moves
    """


    def __init__(self, width, height):
        self.shape = (width, height)


    def collision(self, x, y, snake, opponent):
        valid = (
            0 <= x < self.shape[0] and
            0 <= y < self.shape[1] and
            (x, y) not in snake[1:] and
            (x, y) not in opponent
        )

        return not valid

    def predict(self, snake, food, opponent=()):
        """

        Params
        ------
        snake : list of tuples
            List containing the locations of snake.
            Last element in the list is head, rest are tails
        food : tuple
            Location of food.
        opponent : list of tuples
            List containing the locations of the opponent snake (optional)

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

            if not self.collision(new_dir[0], new_dir[1], snake, opponent):  # check collision
                min_dist = distance(new_dir, food)
                dists.append((min_dist, idx))
        
        dists.sort()
        if len(dists) > 0:
            return dists[0][1]
        else:
            return 0

class MinimaxSnakeAgent:
    """
    Class representing an FSM based agent


    Methods
    -------
    predict(list, tuple):
        Given a list containing snake's location, choose a direction for snake

    
    """

    def __init__(self, width, height):
        self.shape = (width, height)

    def collision(self, x, y, snake, opponent, turn=0):
        if turn == 0:   # turn aka. who is moving
            valid = (
                0 <= x < self.shape[0] and
                0 <= y < self.shape[1] and
                (x, y) not in snake and
                (x, y) not in opponent
            )
        elif turn == 1:
            valid = (
                0 <= x < self.shape[0] and
                0 <= y < self.shape[1] and
                (x, y) not in snake and
                (x, y) not in opponent[1:]
            )

        return not valid
    
    def predict(self, snake, food, opponent):
        """

        Params
        ------
        snake : list of tuples
            List containing the locations of snake.
            Last element in the list is head, rest are tails
        food : tuple
            Location of food.
        opponent : list of tuples
            List containing the locations of the opponent snake (required for minimax agent)

        Returns
        -------
        int :
            Direction of snake
        """

        scores = []
        for idx in range(4):
            # score all possible moves

            x = snake[-1][0] + move_data[idx][0]
            y = snake[-1][1] + move_data[idx][1]

            if not self.collision(x, y, snake, opponent):
                # transition state and run minimax
                state = snake.copy()
                state.pop(0)
                state.append((x, y))

                score = self.minimax(state, food, opponent, player=1)
                scores.append((idx, score))
        
        if len(scores) > 0:
            scores.sort(key= lambda x: x[1])
            return scores[-1][0]
        return 0
    
    def minimax(self, snake, food, opponent, player=0, depth=0, limit=8):
        # estimate future score for a given state
        # evaluate until food is found or collision happens

        penalty = .98 ** depth  # prioritize more efficient paths

        if snake[-1] == food:
            return 10 * (self.calculate_reachable_space(snake, opponent) - 5) * penalty    # max score 20 based on reachable space
        elif opponent[-1] == food:
            return -10 * (self.calculate_reachable_space(opponent, snake) - 5) * penalty
        
        # if recursion limit reached, estimate the winner
        if not depth < limit:
            # score based on who is closer to goal
            distance1 = distance(snake[-1], food)
            distance2 = distance(opponent[-1], food)

            return self.calculate_reachable_space(snake, opponent) / distance1 -  self.calculate_reachable_space(opponent, snake) / distance2

        agent = snake if player == 0 else opponent
        states = []

        # Iterate through the move_data array [up, right, down, left]
        # up to 3 valid moves, branching factor = 3
        for idx in range(4):
            # head + direction = new head x
            new_dir = (agent[-1][0] + move_data[idx][0], agent[-1][1] + move_data[idx][1])
            
            if not self.collision(new_dir[0], new_dir[1], snake, opponent, player):
                # generate transition states
                state = agent.copy()
                state.pop(0)
                state.append(new_dir)
                states.append(state)
        

        if len(states) == 0:
            # agent has no valid next move
            return -100 * len(opponent) * penalty if player == 0 else 100 * len(snake) * penalty
        else:
            scores = []

            for state in states:
                # track game states to prevent cycles and avoid re-exploring
                score = self.minimax(state, food, opponent, 1, depth + 1) if player == 0 else self.minimax(snake, food, state, 0, depth + 1)
                scores.append(score)

            scores.sort()

            return scores[-1] if player == 0 else scores[0]
    
    def calculate_reachable_space(self, snake, opponent, limit=10):
        visited = set()
        q = deque()
        q.append(snake[-1])
        visited.add(snake[-1])

        while q and len(visited) < limit:
            node = q.popleft()

            for x, y in move_data:
                new_dir = (node[0] + x, node[1] + y)

                if 0 <= new_dir[0] < self.shape[0]:
                    if 0 <= new_dir[1] < self.shape[1]:
                        if new_dir not in visited and new_dir not in snake and new_dir not in opponent:
                            visited.add(new_dir)
                            q.append(new_dir)

        return len(visited) - 1     # substract 1 to account for head