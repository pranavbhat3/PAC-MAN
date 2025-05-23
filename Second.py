import pygame
import sys
import csv
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random
from collections import deque

move_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
inv_move_map = {v: k for k, v in move_map.items()}
ghost_move_delay = 250
# --- Maze Definition ---
predefined_maze = [
    "###############################",
    "#........##.......##........#",
    "#.####.#.##.#####.##.#.####.#",
    "#.#  #.#.##.#   #.##.#.#  #.#",
    "#.####.#.##.#####.##.#.####.#",
    "#...........................#",
    "#.####.#####.#.#####.#######",
    "#.#  #.#   #.#.#   #.#     #",
    "#.####.#####.#.#####.#.###.#",
    "#...............P........G.#",
    "###############################"
]

# Convert to 2D array
maze = [list(row) for row in predefined_maze]

# Maze dimensions
MAZE_WIDTH = len(maze[0])
MAZE_HEIGHT = len(maze)
TILE_SIZE = 32
WIDTH = MAZE_WIDTH * TILE_SIZE
HEIGHT = MAZE_HEIGHT * TILE_SIZE

# --- Pygame init ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pac-Man with Adaptive Ghost AI")
font_input = pygame.font.SysFont(None, 48)
font_countdown = pygame.font.SysFont(None, 200)
font_game = pygame.font.SysFont(None, 36)
clock = pygame.time.Clock()

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREY = (100, 100, 100)

# --- Maze & Game Logic ---
def find_tile(tile_char):
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == tile_char:
                return (x, y)
    return (1, 1)

def find_free_positions():
    free_positions = []
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == '.' or cell == ' ':
                free_positions.append((x,y))
    return free_positions

def manhattan_dist(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

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

def valid_move(pos):
    x, y = pos
    if 0 <= y < len(maze) and 0 <= x < len(maze[y]):
        return maze[y][x] not in ['#']
    return False

def move(pos, direction):
    x, y = pos
    if direction == "LEFT":
        new_pos = (x - 1, y)
    elif direction == "RIGHT":
        new_pos = (x + 1, y)
    elif direction == "UP":
        new_pos = (x, y - 1)
    elif direction == "DOWN":
        new_pos = (x, y + 1)
    else:
        new_pos = (x, y)
    return new_pos if valid_move(new_pos) else pos

def get_username():
    username = ""
    input_active = True
    while input_active:
        screen.fill(BLACK)
        draw_text("Enter username (bunny/owl):", (WIDTH // 2, 100), font_input)
        draw_text(username, (WIDTH // 2, HEIGHT // 2), font_input)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if username.lower() in ["bunny", "owl"]:
                        input_active = False
                    else:
                        username = ""
                elif event.key == pygame.K_BACKSPACE:
                    username = username[:-1]
                else:
                    if len(username) < 10 and event.unicode.isalpha():
                        username += event.unicode
        clock.tick(30)
    return username.lower()

def countdown_timer(seconds=3):
    for i in range(seconds, 0, -1):
        screen.fill(BLACK)
        draw_text(str(i), (WIDTH // 2, HEIGHT // 2), font_countdown)
        pygame.display.flip()
        pygame.time.delay(1000)

# --- Classes ---
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

# --- Ghost AI ---
def ghost_bfs_move(ghost_pos, player_pos):
    queue = deque([(ghost_pos, [])])
    visited = set([ghost_pos])
    while queue:
        current_pos, path = queue.popleft()
        if current_pos == player_pos:
            return path[0] if path else current_pos
        x, y = current_pos
        for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
            if 0 <= ny < len(maze) and 0 <= nx < len(maze[0]) and valid_move((nx, ny)) and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
    return ghost_pos

def move_ghost_bfs(ghost, player_pos):
    ghost.pos = ghost_bfs_move(ghost.pos, player_pos)

# --- Model Training ---
def collect_training_data(player_name, features, label):
    filename = f"training_data_{player_name}.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(list(features.keys()) + ["label"])
        writer.writerow(list(features.values()) + [label])

def train_model(player_name):
    filename = f"training_data_{player_name}.csv"
    if not os.path.isfile(filename):
        return None
    data = pd.read_csv(filename)
    if data.empty or "label" not in data.columns:
        return None
    X = data.drop("label", axis=1)
    y = data["label"].map(move_map)
    if y.isnull().any():
        return None
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    return clf

def features_for_model(player_pos, ghost_pos, last_move):
    px, py = player_pos
    gx, gy = ghost_pos
    dx = px - gx
    dy = py - gy
    lm = move_map.get(last_move, 0)
    return {
        "px": px, "py": py,
        "gx": gx, "gy": gy,
        "dx": dx, "dy": dy,
        "last_move": lm
    }

# --- Game Setup ---
player_start = find_tile("P")
ghost_start = find_tile("G")
maze[player_start[1]][player_start[0]] = '.'
maze[ghost_start[1]][ghost_start[0]] = '.'
player = Player(player_start)
ghost = Ghost(ghost_start)
username = get_username()
countdown_timer()

model = train_model(username)
game_over = False
move_delay = 150
last_move_time = pygame.time.get_ticks()

last_ghost_move_time = pygame.time.get_ticks()

# --- Game Loop ---
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
    if current_time - last_ghost_move_time > ghost_move_delay:
        move_ghost_bfs(ghost, player.pos)
        last_ghost_move_time = current_time
    if player.pos == ghost.pos:
        game_over = True
    player.draw()
    ghost.draw()
    draw_text(f"Player: {username}", (100, HEIGHT - 20), font_game, WHITE)
    pygame.display.flip()
    clock.tick(60)

# --- Game Over ---
screen.fill(BLACK)
draw_text("Game Over!", (WIDTH // 2, HEIGHT // 2 - 50), font_input, RED)
draw_text(f"{username} was caught!", (WIDTH // 2, HEIGHT // 2 + 50), font_input, RED)
pygame.display.flip()
pygame.time.delay(3000)
pygame.quit()
sys.exit()
