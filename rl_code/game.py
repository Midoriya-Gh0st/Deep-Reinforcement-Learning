import random, copy

WIDTH = 8
HEIGHT = 10
MOVES = 25
GEM_COUNT = 4
OBSTACLE_COUNT = 1
OBSTACLE_CHAR = '#'
EMPTY = -1


class Board:
    def __init__(self, rand):
        self._rand = rand
        grid = []  # board
        for _ in range(HEIGHT):
            row = []  # each row
            for _ in range(WIDTH):
                row.append(self._rand.randint(0, GEM_COUNT + OBSTACLE_COUNT - 1))  # randomly generate a gem or obstacle
            grid.append(row)
        self._grid = grid

    def __str__(self):
        return '\n'.join(
            ''.join((' ' if x == EMPTY else (OBSTACLE_CHAR if x >= GEM_COUNT else chr(ord('a') + x))) for x in row)
            for row in self._grid)

    def grid(self):
        # print("1 - game.py - [board] - get grid:", self._grid)
        return self._grid

    @staticmethod
    # drop the gem
    def _drop_one(grid, rand):
        any_dropped = False
        for x in range(WIDTH):
            dropping = False
            for y in range(HEIGHT - 1, -1, -1):
                c = grid[y][x]
                dropping = dropping or c == EMPTY
                if dropping and y > 0:
                    grid[y][x] = grid[y - 1][x]  # up gem will drop
            if dropping:  # drop if below is null
                grid[0][x] = rand.randint(0, GEM_COUNT + OBSTACLE_COUNT - 1)
            any_dropped = any_dropped or dropping
        return any_dropped

    @staticmethod
    # judge when to clear
    def _check_clear_obstacle(x, y, grid, to_clear):
        if 0 <= x < WIDTH and 0 <= y < HEIGHT and grid[y][x] >= GEM_COUNT:
            to_clear.append((x, y))  # a list stores xy that will be cleared

    @staticmethod
    def _match(grid):
        clear = []
        score = 0

        # Identify matches
        # calculate score
        limits = (WIDTH, HEIGHT)
        for dir in [(1, 0), (0, 1)]:
            ortho = (1 - dir[0], 1 - dir[1])
            start = (0, 0)
            while start[0] < limits[0] and start[1] < limits[1]:
                run = 0
                prev = EMPTY
                cur = start
                while cur[0] <= limits[0] and cur[1] <= limits[1]:
                    c = grid[cur[1]][cur[0]] if (cur[0] < limits[0] and cur[1] < limits[1]) else EMPTY
                    if c == prev:
                        run += 1
                    else:
                        if prev < GEM_COUNT and run >= 3:
                            score += (min(run, 5) - 1) * (min(run, 5) + 1)
                            overwrite = (cur[0] - run * ortho[0], cur[1] - run * ortho[1])
                            while overwrite[0] < cur[0] or overwrite[1] < cur[1]:
                                clear.append(overwrite)
                                overwrite = (overwrite[0] + ortho[0], overwrite[1] + ortho[1])
                        prev = c
                        run = 1
                    cur = (cur[0] + ortho[0], cur[1] + ortho[1])
                start = (start[0] + dir[0], start[1] + dir[1])

        # Delete "empty"s
        clear_obstacles = []
        for x, y in clear:
            grid[y][x] = EMPTY
            Board._check_clear_obstacle(x + 1, y, grid, clear_obstacles)
            Board._check_clear_obstacle(x - 1, y, grid, clear_obstacles)
            Board._check_clear_obstacle(x, y + 1, grid, clear_obstacles)
            Board._check_clear_obstacle(x, y - 1, grid, clear_obstacles)

        for x, y in clear_obstacles:
            grid[y][x] = EMPTY
            # print("1 - game.py - [Board] - empty(y,x):", y, x)

        return score

    @staticmethod
    # check match
    def _step_impl(grid, rand):
        """ returns (anything_changed, score_delta) """
        any_dropped = Board._drop_one(grid, rand)
        if any_dropped:
            # print("1 - game.py - [Board] - step - Drop!!! <<<")
            return True, 0
        else:  # if no drop
            sdif = Board._match(grid)  # sdif = delta reward
            return sdif != 0, sdif

    @staticmethod
    # the move operation - logic
    def _move_in_place(grid, move):
        """Returns True if successful, False otherwise"""
        x, y, vert = move
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return False
        if vert:  # vertical
            xp, yp = x, y + 1  # notice the coord O(0,0) is at top left
        else:  # hori
            xp, yp = x + 1, y
        if xp >= WIDTH or yp >= HEIGHT:
            return False
        grid[y][x], grid[yp][xp] = grid[yp][xp], grid[y][x]  # exchange two
        return True

    @staticmethod
    def _matches_anything(grid, move):
        grid_prime = copy.deepcopy(grid)  # asbolute new, will not impact the origianl
        if Board._move_in_place(grid_prime, move):
            chg, sdif = Board._step_impl(grid_prime, random)
            if sdif > 0:
                # print("1 - game.py - [match] - yes! match:", sdif)
                pass
            return sdif > 0  # if some "drop"s, -> match yes -> get sdif(reward)
        else:
            return False

    @staticmethod
    def matches_something(grid, move):  # just a copy
        q = Board._matches_anything(grid, move)
        # print("1 - game.py - [match] - this try, match:", q)
        return q

    def move(self, move):  # ****  IMPORTANT <- pygame.env.step(action) here the move is the actually move
        return Board._move_in_place(self._grid, move)

    def step(self):
        return Board._step_impl(self._grid, self._rand)

    def matching_moves(self):
        ret = []
        for x in range(WIDTH):
            for y in range(HEIGHT):
                for d in [False, True]:  # direction
                    mv = (x, y, d)
                    if Board._matches_anything(self._grid, mv):
                        ret.append(mv)
        return ret


class GameLogic:
    def __init__(self, seed=None):  # init the logic
        if seed is None:
            seed = random.getrandbits(128)
        rand = random.Random(seed)
        self._moves_left = MOVES  # moves left
        self._score = 0  # current score 
        self._board = Board(rand)  # new game board
        changes = True
        while changes:  # settle
            changes, _ = self._board.step()

    def score(self):  # get score
        return self._score

    def is_gameover(self):  # check gameover (moves_left=0)
        return self._moves_left <= 0

    def matching_moves(self):
        return self._board.matching_moves()

    def board(self):  # string format
        return str(self._board)

    def grid(self):  # not used
        return self._board.grid()

    def moves_left(self):
        return self._moves_left

    def play(self, move):  # execute an action that got by prob <- (env.step(action))
        # print("1 - game.py - [run] - play >>> STR >>>:", '-'*20)
        interm = []
        sdif = 0
        if not self.is_gameover():
            self._board.move(move)
            changes = True
            while changes:
                interm.append(str(self._board))
                changes, delta = self._board.step()  # if anydrop=1, return true, go on
                sdif += delta
            interm = interm[:-1]  # get the last state 
            if sdif == 0:  # bad move 
                # print("1 - game.py - [run] - play: Invalid Move!")
                sdif = -5
            else:
                # print("1 - game.py - [run] - play: Great Move!")
                pass
            self._score += sdif
        self._moves_left -= 1
        # print("1 - game.py - [run] - play - interm:", interm)  # (IM为[])
        # print("1 - game.py - [run] - play: get sdif:", sdif)
        # print("1 - game.py - [run] - play <<< END <<<:", '-' * 20)
        return str(self._board), sdif, self.is_gameover(), interm
        # bad move can be executed but will get penalty (i.e. will impact the state (the board))
