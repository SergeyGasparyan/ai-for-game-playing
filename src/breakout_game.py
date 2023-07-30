import pygame as pg

SCREEN_SIZE = 424, 430

BRICK_HEIGHT, BRICK_WIDTH = 13, 32

PADDLE_HEIGHT, PADDLE_WIDTH = 8, 50
PADDLE_Y = SCREEN_SIZE[1] - PADDLE_HEIGHT - 10
MAX_PADDLE_X = SCREEN_SIZE[0] - PADDLE_WIDTH

BALL_DIAMETER = 12
BALL_RADIUS = BALL_DIAMETER // 2
MAX_BALL_X = SCREEN_SIZE[0] - BALL_DIAMETER
MAX_BALL_Y = SCREEN_SIZE[1] - BALL_DIAMETER

NUM_BRICKS_VERTICAL = 11
NUM_BRICKS_HORIZONTAL = 12

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
PADDLE_COLOR = (129, 133, 137)
BRICK_COLOR = (153, 255, 204)

FPS = 60

pg.init()
screen = pg.display.set_mode(SCREEN_SIZE)
pg.display.set_caption("Breakout - Atari")
clock = pg.time.Clock()


class Breakout:
    def __init__(self, vel=12):
        self.capture = 0
        self.vel = vel
        self.ball_vel = [vel, -vel]
        self.reward = 0.1
        self.terminal = False

        self.paddle = pg.Rect(
            SCREEN_SIZE[1] // 2, PADDLE_Y, PADDLE_WIDTH, PADDLE_HEIGHT
        )
        self.ball = pg.Rect(
            SCREEN_SIZE[1] // 2 + 10,
            PADDLE_Y - BALL_DIAMETER,
            BALL_DIAMETER,
            BALL_DIAMETER,
        )
        self.create_bricks()

    def create_bricks(self):
        self.bricks = []
        y_ofs = 20
        for _ in range(NUM_BRICKS_VERTICAL):
            x_ofs = 15

            for _ in range(NUM_BRICKS_HORIZONTAL):
                self.bricks.append(pg.Rect(x_ofs, y_ofs, BRICK_WIDTH, BRICK_HEIGHT))
                x_ofs += BRICK_WIDTH + 1

            y_ofs += BRICK_HEIGHT + 1

    def draw_bricks(self):
        for i, brick in enumerate(self.bricks):
            pg.draw.rect(screen, BRICK_COLOR, brick)

    def draw_paddle(self):
        pg.draw.rect(screen, PADDLE_COLOR, self.paddle)

    def draw_ball(self):
        pg.draw.circle(
            screen,
            WHITE,
            (self.ball.left + BALL_RADIUS, self.ball.top + BALL_RADIUS),
            BALL_RADIUS,
        )

    def check_input(self, input_action):
        # 0: LEFT, 1: Right
        if input_action[0] == 1:
            self.paddle.left -= self.vel
            if self.paddle.left < 0:
                self.paddle.left = 0

        elif input_action[1] == 1:
            self.paddle.left += self.vel
            if self.paddle.left > MAX_PADDLE_X:
                self.paddle.left = MAX_PADDLE_X

    def move_ball(self):
        self.ball.left += self.ball_vel[0]
        self.ball.top += self.ball_vel[1]

        if self.ball.left <= 0:
            self.ball.left = 0
            self.ball_vel[0] = -self.ball_vel[0]
        elif self.ball.left >= MAX_BALL_X:
            self.ball.left = MAX_BALL_X
            self.ball_vel[0] = -self.ball_vel[0]

        if self.ball.top < 0:
            self.ball.top = 0
            self.ball_vel[1] = -self.ball_vel[1]
        elif self.ball.top >= MAX_BALL_Y:
            self.ball.top = MAX_BALL_Y
            self.ball_vel[1] = -self.ball_vel[1]

    def take_action(self, input_action):
        pg.event.pump()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        screen.fill(BLACK)
        self.check_input(input_action)
        self.move_ball()

        # Handle Collisions
        for brick in self.bricks:
            if self.ball.colliderect(brick):
                self.reward = 2
                self.ball_vel[1] = -self.ball_vel[1]
                self.bricks.remove(brick)
                break

        if len(self.bricks) == 0:
            self.terminal = True
            self.__init__()

        if self.ball.colliderect(self.paddle):
            self.ball.top = PADDLE_Y - BALL_DIAMETER
            self.ball_vel[1] = -self.ball_vel[1]
        elif self.ball.top > self.paddle.top:
            self.terminal = True
            self.__init__()
            self.reward = -2

        self.draw_bricks()
        self.draw_ball()
        self.draw_paddle()

        image_data = pg.surfarray.array3d(pg.display.get_surface())

        pg.display.update()
        clock.tick(FPS)

        return image_data, self.reward, self.terminal
