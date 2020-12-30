import math, random


# CHANGE THESE TO MAKE THE GAME RUN SLOWER OR FASTER:
import game

ACC = 1
TIME_PER_SWAP = 0.35 / ACC
TIME_PER_MATCH = 0.35 / ACC
TIME_PER_DROP = 0.075 / ACC
assert TIME_PER_SWAP > 0 and TIME_PER_MATCH > 0 and TIME_PER_DROP > 0

WAIT_AFTER_MOVE = 0.5 / ACC
WAIT_AFTER_GAME = 2.0 / ACC

try:
    import pygame

    WORKS = True
except:
    print()
    print(' *** WARNING ***')
    print()
    print('Please install the pygame package!')
    print('To install for the current user only, try:')
    print('\tpip install -U pygame --user')
    print('To install globally, try:')
    print('\tpip install pygame')
    print()
    WORKS = False

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
HASHCOLOR = (100, 100, 100)
GEM_SIZE = 72
GEM_INSET = 3
SPACING = 4
TEXT_SIZE = 20
TEXT_SIZE_SMALL = 12

font_path_0 = 'LiberationMono-Regular.ttf'
font_path_1 = '../input/cs4287files/LiberationMono-Regular.ttf'  # 'LiberationMono-Regular.ttf'
font_path_2 = './drive/MyDrive/Colab Notebooks/CS4287-FPRJ/LiberationMono-Regular.ttf'

pygame.init()
pygame.display.set_caption('DENDY RUSH')
_font = pygame.font.Font(font_path_0, TEXT_SIZE)
_font_small = pygame.font.Font(font_path_0, TEXT_SIZE_SMALL)
_font_small.set_bold(True)

ST_READY = 0
ST_ANIMATING = 1
ST_POSTGAME = 2

TRANS_SWAP = 0
TRANS_DISSOLVE = 1
TRANS_DROP = 2


def _alpha_mul(color, alpha):
    return 255 - alpha * (255 - color[0]), 255 - alpha * (255 - color[1]), 255 - alpha * (255 - color[2])


def _ngon(surface, center, radius, n, color):
    verts = []
    for i in range(n):
        angle = i / n * 2 * math.pi
        x = center[0] + radius * math.sin(angle)
        y = center[1] + radius * math.cos(angle)
        verts.append((x, y))
    pygame.draw.polygon(surface, color, verts)


def _draw_text(surface, position, text, font, alignment=(0.0, 0.0), color=(0, 0, 0), background=None):
    tsurf = font.render(text, True, color, background)
    rect = tsurf.get_rect()
    sz = (rect.width, rect.height)
    rect = rect.move((position[0] - alignment[0] * sz[0], position[1] - alignment[1] * sz[1]))
    surface.blit(tsurf, rect)


def _draw_gem(surface, center, gem, alpha):
    rad = 0.5 * (GEM_SIZE - GEM_INSET)
    if gem == '#':
        color = _alpha_mul(HASHCOLOR, alpha)
        pygame.draw.rect(surface, color,
                         pygame.Rect(center[0] - rad, center[1] - rad, GEM_SIZE - GEM_INSET, GEM_SIZE - GEM_INSET))
    elif gem == ' ':
        color = (255,) * 3
    else:
        d = ord(gem) - ord('a')
        color = _alpha_mul(COLORS[d], alpha)
        _ngon(surface, center, rad, d + 3, color)
    return color


def _gem_pos(p):
    x, y = p
    rad = 0.5 * (GEM_SIZE - GEM_INSET)
    return (SPACING + GEM_SIZE * x + rad, SPACING + GEM_SIZE * y + rad)


def _interpolate(a, b, t):
    ax, ay = a
    bx, by = b
    omt = 1 - t
    return (ax * t + bx * omt, ay * t + by * omt)


class Game:
    def __init__(self, ai_callback, transition_callback, end_of_game_callback, speed=1.0, seed=None):
        if seed == None:
            seed = random.getrandbits(128)
        print('Seed:', seed)
        self.rand = random.Random(seed)
        self.message = ''
        self.screen = pygame.display.set_mode(
            (game.WIDTH * GEM_SIZE + 3 * SPACING + 8 * TEXT_SIZE, game.HEIGHT * GEM_SIZE + 3 * SPACING + TEXT_SIZE))
        self.ai_callback = ai_callback
        self.transition_callback = transition_callback
        self.end_of_game_callback = end_of_game_callback
        self.clock = pygame.time.Clock()
        self.speed = speed
        self._new_game()

    # not the "RL state", here the state is about the programme operation
    def _enter_state(self, state):
        self.state = state
        self.time_in_state = 0
        if state == ST_ANIMATING:
            self.anim_idx = self.anim_t = 0

    def _ask_ai(self):  # callback ai_callback()
        # print("2 - gra.py - [game] - _ask_ai")
        before = self.game_logic.board()  # 获取上一步棋盘 (state, 状态?)
        # input: state(board), reward(score), moves_left
        # !!! IMPORATNT ask ai to get "action"
        move = self.ai_callback(before, self.game_logic.score(), self.game_logic.moves_left())
        # output: move (x, y, dire)
        nxt, sdif, _, interm = self.game_logic.play(move)  # true execution

        self.board_history.append(nxt)
        self.move_history.append(move)
        self.score_history.append(self.game_logic.score())
        self.transition_callback(before, move, sdif, nxt, self.game_logic.moves_left())
        self.lastmove = move
        self.anim_states = [before] + interm + [nxt]
        self.anim_t = 0
        self.anim_idx = 0
        self._enter_state(ST_ANIMATING)

    def _animate(self, delta):
        frm = self._from_state().split('\n')
        to = self._to_state().split('\n')
        tp, _ = Game._analyze_transition(frm, to, self.anim_idx == 0, self.lastmove)
        if tp == TRANS_DISSOLVE:
            duration = TIME_PER_MATCH
        elif tp == TRANS_SWAP:
            duration = TIME_PER_SWAP
        else:
            assert tp == TRANS_DROP
            duration = TIME_PER_DROP
        t = self.anim_t + delta / duration
        self.anim_idx += int(t)
        self.anim_t = t % 1
        if self.anim_idx >= len(self.anim_states) - 1:
            self.anim_idx = len(self.anim_states) - 1
            self.anim_t = 0
            self.displayscore = self.game_logic.score()
            self.displaytext = self.game_logic.board()
            self._enter_state(ST_READY)

    @staticmethod
    def _analyze_transition(frm, to, isfirst, lastmove):
        if isfirst:
            xd = 0 if lastmove[2] else 1
            swapped = [(lastmove[0], lastmove[1]), (lastmove[0] + xd, lastmove[1] + 1 - xd)]
            return TRANS_SWAP, swapped
        else:
            frmcnt = 0
            tocnt = 0
            for row in frm:
                for c in row:
                    if c == ' ':
                        frmcnt += 1
            for row in to:
                for c in row:
                    if c == ' ':
                        tocnt += 1
            if tocnt > frmcnt:
                return TRANS_DISSOLVE, None
            else:
                drop_start = []
                for x in range(game.WIDTH):
                    st = -1
                    for y in range(game.HEIGHT - 1, -1, -1):
                        if frm[y][x] == ' ':
                            st = y
                            break
                    drop_start.append(st)
                return TRANS_DROP, drop_start

    def _from_state(self):
        return self.anim_states[self.anim_idx]

    def _to_state(self):
        return self.anim_states[self.anim_idx + 1] if self.anim_idx < len(self.anim_states) - 1 else self.anim_states[
            len(self.anim_states) - 1]

    def _render_game(self):
        frm = self._from_state().split('\n')
        to = self._to_state().split('\n')
        trans, tinfo = Game._analyze_transition(frm, to, self.anim_idx == 0, self.lastmove)
        for y, row in enumerate(to):
            for x, c in enumerate(row):
                pos = _gem_pos((x, y))
                alpha = 1
                if trans == TRANS_SWAP:
                    if (x, y) == tinfo[0]:
                        pos = _interpolate(_gem_pos((x, y)), _gem_pos(tinfo[1]), self.anim_t)
                    elif (x, y) == tinfo[1]:
                        pos = _interpolate(_gem_pos((x, y)), _gem_pos(tinfo[0]), self.anim_t)
                elif trans == TRANS_DISSOLVE:
                    if c != frm[y][x]:
                        alpha = 1 - self.anim_t
                        c = frm[y][x]
                else:
                    assert trans == TRANS_DROP
                    if y <= tinfo[x]:
                        pos = pos[0], pos[1] - (1 - self.anim_t) * GEM_SIZE
                _draw_gem(self.screen, pos, c, alpha)
                textcol = (255 - alpha * 255,) * 3
                _draw_text(self.screen, pos, "%d, %d" % (x, y), _font_small, (0.5, 0.5), textcol)

    def _render_text(self):
        _draw_text(self.screen, (SPACING, 2 * SPACING + GEM_SIZE * game.HEIGHT), self.message, _font)
        _draw_text(self.screen, (SPACING + GEM_SIZE * game.WIDTH, SPACING), 'Moves: %d' % self.game_logic.moves_left(), _font)
        _draw_text(self.screen, (SPACING + GEM_SIZE * game.WIDTH, SPACING + TEXT_SIZE), 'Score: %d' % self.displayscore,
                   _font)
        text = 'As text:\n\n' + self.displaytext
        splt = text.split('\n')
        sz = self.screen.get_size()
        for i, ln in enumerate(splt):
            pos = (sz[0] - SPACING, sz[1] - SPACING - (len(splt) - i) * TEXT_SIZE)
            _draw_text(self.screen, pos, ln, _font, (1, 0))

    def _render_postgame(self):
        sz = self.screen.get_size()
        pygame.draw.rect(self.screen, (50, 50, 50), pygame.Rect(0.25 * sz[0], 0.25 * sz[1], 0.5 * sz[0], 0.5 * sz[1]))
        _draw_text(self.screen, (0.5 * sz[0], 0.5 * sz[1] - 0.5 * TEXT_SIZE_SMALL), 'GAME OVER', _font, (0.5, 1.0),
                   (255, 255, 255))
        _draw_text(self.screen, (0.5 * sz[0], 0.5 * sz[1] + 0.5 * TEXT_SIZE_SMALL), 'Score: %d' % self.game_logic.score(),
                   _font, (0.5, 0.0), (255, 255, 255))

    def _new_game(self):
        self.game_logic = game.GameLogic(self.rand.getrandbits(128))  # game logic
        self.anim_idx = 1
        self.anim_t = 0.0
        board = self.game_logic.board()
        self.anim_states = [board] * 2
        self.lastmove = None
        self.board_history = [board]
        self.move_history = []
        self.score_history = [0]
        self.displayscore = 0
        self.displaytext = self.game_logic.board()
        self._enter_state(ST_READY)

    def reset(self):  # simulate the reset of env (env.reset())
        self._new_game()
        return self.board_history[-1]

    def run(self):
        keepRunning = True
        while keepRunning:
            self.clock.tick()
            delta = self.clock.get_time() * 0.001 * self.speed
            self.time_in_state += delta

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    keepRunning = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        keepRunning = False

            if self.state == ST_READY:  # if ready, call action
                if self.time_in_state > WAIT_AFTER_MOVE:
                    if self.game_logic.is_gameover():
                        # print("3 - gra.py - [run] - r-tis-wam - game over")
                        self.play_another = self.end_of_game_callback(self.board_history, self.score_history,
                                                                      self.move_history, self.game_logic.score())
                        self._enter_state(ST_POSTGAME)
                    else:
                        # print("3 - gra.py - [run] - r-tis-wam - _ask_ai() >>> STR >>>")
                        self._ask_ai()
                        # print("3 - gra.py - [run] - r-tis-wam - current board:\n", self.board_history[-1])
                        # print("3 - gra.py - [run] - r-tis-wam - _ask_ai() <<< END <<<")

            elif self.state == ST_ANIMATING:
                # print("3 - gra.py - [run] - state == ST_ANIMATING")
                self._animate(delta)
            else:
                assert self.state == ST_POSTGAME
                # print("3 - gra.py - [run] - state == ST_POSTGAME") # should show aftet game over
                if self.time_in_state >= WAIT_AFTER_GAME:
                    if self.play_another:
                        # print("3 - gra.py - [run] - _new_game()")
                        self._new_game()
                    else:
                        keepRunning = False

            self.screen.fill((255, 255, 255))
            self._render_game()
            self._render_text()
            if self.state == ST_POSTGAME:
                self._render_postgame()
            pygame.display.flip()
