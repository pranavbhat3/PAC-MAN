import pygame
import sys
import csv
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random
from collections import deque

# --- Config ---
move_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
inv_move_map = {v: k for k, v in move_map.items()}
ghost_move_delay = 250
data_filename_template = "ghost_data_{}.csv"

# --- Maze Definition ---
maze_layout = [
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#.......P..................#",
    "#.####.##.########.##.####.#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.##### ## #####.######",
    "######.##### ## #####.######",
    "######.##          ##.######",
    "######.## ######## ##.######",
    "######.## ######## ##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#...##................##...#",
    "###.##.##.########.##.##.###",
    "###.##.##.########.##.##.###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#.##########.##.##########.#",
    "#..............G...........#",
    "############################"
]
maze = [list(row) for row in maze_layout]

# --- Pygame Init ---
MAZE_WIDTH = len(maze[0])
MAZE_HEIGHT = len(maze)
TILE_SIZE = 32
WIDTH = MAZE_WIDTH * TILE_SIZE
HEIGHT = MAZE_HEIGHT * TILE_SIZE

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pac-Man with Adaptive Ghost AI")
font_input = pygame.font.SysFont(None, 48)
font_countdown = pygame.font.SysFont(None, 200)
font_game = pygame.font.SysFont(None, 36)
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# --- Maze Utils ---
score = 0
dots = set()
for y, row in enumerate(maze):
    for x, tile in enumerate(row):
        if tile == '.' or tile == ' ':
            dots.add((x, y))

def find_tile(tile_char):
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == tile_char:
                return (x, y)
    return (1, 1)

def draw_text(text, pos, font, color=WHITE):
    txt_surface = font.render(text, True, color)
    rect = txt_surface.get_rect(center=pos)
    screen.blit(txt_surface, rect)

def draw_maze():
    for y, row in enumerate(maze):
        for x, tile in enumerate(row):
            rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if tile == '#':
                pygame.draw.rect(screen, BLUE, rect)
            else:
                pygame.draw.rect(screen, BLACK, rect)
            if (x, y) in dots:
                pygame.draw.circle(screen, YELLOW, rect.center, 4)

def valid_move(pos):
    x, y = pos
    return 0 <= y < MAZE_HEIGHT and 0 <= x < MAZE_WIDTH and maze[y][x] != '#'

def move(pos, direction):
    x, y = pos
    if direction == "LEFT": x -= 1
    elif direction == "RIGHT": x += 1
    elif direction == "UP": y -= 1
    elif direction == "DOWN": y += 1
    return (x, y) if valid_move((x, y)) else pos

def get_username():
    username = ""
    input_active = True
    while input_active:
        screen.fill(BLACK)
        draw_text("Enter username (bunny/owl):", (WIDTH // 2, 100), font_input)
        draw_text(username, (WIDTH // 2, HEIGHT // 2), font_input)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and username.lower() in ["bunny", "owl"]:
                    input_active = False
                elif event.key == pygame.K_BACKSPACE:
                    username = username[:-1]
                elif len(username) < 10 and event.unicode.isalpha():
                    username += event.unicode
        clock.tick(30)
    return username.lower()

def countdown_timer(seconds=3):
    for i in range(seconds, 0, -1):
        screen.fill(BLACK)
        draw_text(str(i), (WIDTH // 2, HEIGHT // 2), font_countdown)
        pygame.display.flip()
        pygame.time.delay(1000)

class Player:
    def __init__(self, pos):
        self.pos = pos
        self.last_move = "UP"
    def move(self, direction):
        new_pos = move(self.pos, direction)
        if new_pos != self.pos:
            self.pos = new_pos
            self.last_move = direction
    def draw(self):
        rect = pygame.Rect(self.pos[0]*TILE_SIZE, self.pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.circle(screen, YELLOW, rect.center, TILE_SIZE//2 - 4)

class Ghost:
    def __init__(self, pos):
        self.pos = pos
    def draw(self):
        rect = pygame.Rect(self.pos[0]*TILE_SIZE, self.pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.circle(screen, RED, rect.center, TILE_SIZE//2 - 4)

# Ghost AI movement using prediction or BFS
def predict_player_move(model, ghost_pos, player_pos):
    features = np.array([[player_pos[0], player_pos[1], ghost_pos[0], ghost_pos[1]]])
    pred = model.predict(features)[0]
    return inv_move_map[pred]

def ghost_ai_move(ghost, player_pos, model):
    move_dir = predict_player_move(model, ghost.pos, player_pos)
    ghost.pos = move(ghost.pos, move_dir)

def ghost_bfs_move(ghost_pos, player_pos):
    queue = deque([(ghost_pos, [])])
    visited = set([ghost_pos])
    while queue:
        current_pos, path = queue.popleft()
        if current_pos == player_pos:
            return path[0] if path else current_pos
        x, y = current_pos
        for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
            if valid_move((nx, ny)) and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
    return ghost_pos

# Setup
player_start = find_tile("P")
ghost_start = find_tile("G")
maze[player_start[1]][player_start[0]] = '.'
maze[ghost_start[1]][ghost_start[0]] = '.'
player = Player(player_start)
ghost = Ghost(ghost_start)
username = get_username()
countdown_timer()
model = None
log = []
csv_file = data_filename_template.format(username)
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    if len(df) >= 20:
        X = df[["px", "py", "gx", "gy"]]
        y = df["move"]
        model = DecisionTreeClassifier().fit(X, y)

# Main Game Loop
game_over = False
move_delay = 150
last_move_time = pygame.time.get_ticks()
last_ghost_move_time = pygame.time.get_ticks()

while not game_over:
    screen.fill(BLACK)
    draw_maze()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    player_move = None
    if keys[pygame.K_UP]: player_move = "UP"
    elif keys[pygame.K_DOWN]: player_move = "DOWN"
    elif keys[pygame.K_LEFT]: player_move = "LEFT"
    elif keys[pygame.K_RIGHT]: player_move = "RIGHT"

    current_time = pygame.time.get_ticks()
    if player_move and current_time - last_move_time > move_delay:
        player.move(player_move)
        last_move_time = current_time
        log.append({
            "px": player.pos[0], "py": player.pos[1],
            "gx": ghost.pos[0], "gy": ghost.pos[1],
            "move": move_map[player_move]
        })

    if current_time - last_ghost_move_time > ghost_move_delay:
        if model:
            ghost_ai_move(ghost, player.pos, model)
        else:
            ghost.pos = ghost_bfs_move(ghost.pos, player.pos)
        last_ghost_move_time = current_time

    if player.pos == ghost.pos:
        game_over = True

    if player.pos in dots:
        dots.remove(player.pos)
        score += 10

    if not dots:
        screen.fill(BLACK)
        draw_text("You Win!", (WIDTH // 2, HEIGHT // 2 - 50), font_input, YELLOW)
        draw_text(f"Score: {score}", (WIDTH // 2, HEIGHT // 2 + 50), font_input, YELLOW)
        pygame.display.flip()
        pygame.time.delay(3000)
        break

    player.draw()
    ghost.draw()
    draw_text(f"Player: {username}  Score: {score}", (WIDTH // 2, HEIGHT - 20), font_game, WHITE)

    pygame.display.flip()
    clock.tick(60)

# Save movement data
if log:
    df = pd.DataFrame(log)
    if os.path.exists(csv_file):
        old_df = pd.read_csv(csv_file)
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_csv(csv_file, index=False)

# End screen
screen.fill(BLACK)
draw_text("Game Over!", (WIDTH // 2, HEIGHT // 2 - 50), font_input, RED)
draw_text(f"{username} was caught!", (WIDTH // 2, HEIGHT // 2 + 50), font_input, RED)
draw_text(f"Final Score: {score}", (WIDTH // 2, HEIGHT // 2 + 100), font_input, WHITE)
pygame.display.flip()
pygame.time.delay(3000)
pygame.quit()
sys.exit()
