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
# Maze dimensions (odd numbers for proper maze)
MAZE_WIDTH = 31
MAZE_HEIGHT = 23

TILE_SIZE = 32

# Adapt window size to maze dimensions
WIDTH = MAZE_WIDTH * TILE_SIZE
HEIGHT = MAZE_HEIGHT * TILE_SIZE

# --- Pygame init ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pac-Man with Adaptive Ghost AI")

# rest of your code continues as before


font_input = pygame.font.SysFont(None, 48)
font_countdown = pygame.font.SysFont(None, 200)
font_game = pygame.font.SysFont(None, 36)
clock = pygame.time.Clock()

TILE_SIZE = 40
GRID_WIDTH, GRID_HEIGHT = WIDTH // TILE_SIZE, HEIGHT // TILE_SIZE

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREY = (100, 100, 100)

# --- Random Maze Generation ---

import random

def generate_maze_recursive_backtracking(width, height):
    # Ensure odd dimensions for proper maze structure
    if width % 2 == 0:
        width -= 1
    if height % 2 == 0:
        height -= 1

    # Initialize maze with walls
    maze = [['#' for _ in range(width)] for _ in range(height)]

    def carve_passages(cx, cy):
        maze[cy][cx] = '.'  # Mark current cell as passage

        # Define directions: up, down, left, right (2 steps to carve)
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(directions)  # Randomize directions

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            # Check if new position is inside maze bounds and unvisited
            if 1 <= nx < width - 1 and 1 <= ny < height - 1 and maze[ny][nx] == '#':
                # Carve passage between current cell and new cell
                maze[cy + dy // 2][cx + dx // 2] = '.'
                carve_passages(nx, ny)

    # Start carving from (1,1)
    carve_passages(1, 1)

    return maze


# Example usage and printing:
if __name__ == "__main__":
    w, h = 21, 15
    maze = generate_maze_recursive_backtracking(w, h)

    for row in maze:
        print(''.join(row))



def find_free_positions(maze):
    free_positions = []
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == '.':
                free_positions.append((x,y))
    return free_positions

def manhattan_dist(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

def pick_start_positions(maze, min_dist=6):
    free_positions = find_free_positions(maze)
    player_pos = random.choice(free_positions)
    
    candidates = [pos for pos in free_positions if manhattan_dist(pos, player_pos) >= min_dist]
    if not candidates:
        ghost_pos = random.choice(free_positions)
    else:
        ghost_pos = random.choice(candidates)

    return player_pos, ghost_pos

# --- Helper functions ---

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
        return maze[y][x] != '#'
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
    if valid_move(new_pos):
        return new_pos
    return pos

# --- Username Input & Countdown ---

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

# --- Player and Ghost setup ---

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

# --- BFS pathfinding for ghost ---

def ghost_bfs_move(ghost_pos, player_pos, maze):
    queue = deque()
    queue.append((ghost_pos, []))  # (position, path_to_here)
    visited = set()
    visited.add(ghost_pos)

    while queue:
        current_pos, path = queue.popleft()
        if current_pos == player_pos:
            return path[0] if path else current_pos

        x, y = current_pos
        neighbors = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        for nx, ny in neighbors:
            if 0 <= ny < len(maze) and 0 <= nx < len(maze[0]) and maze[ny][nx] != '#' and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
    return ghost_pos

def move_ghost_bfs(ghost, player_pos, maze):
    next_pos = ghost_bfs_move(ghost.pos, player_pos, maze)
    ghost.pos = next_pos

# --- Decision Tree AI ---

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
    move_map_local = {"UP":0, "DOWN":1, "LEFT":2, "RIGHT":3}
    lm = move_map_local.get(last_move, 0)
    return {
        "px": px,
        "py": py,
        "gx": gx,
        "gy": gy,
        "dx": dx,
        "dy": dy,
        "last_move": lm
    }

def ghost_predict_move(model, features):
    if model is None:
        return None
    X_test = np.array([list(features.values())])
    pred = model.predict(X_test)[0]
    return pred

def move_ghost_towards_predicted(ghost, player_pos, predicted_move):
    if predicted_move is None:
        return  # no move
    x, y = ghost.pos
    move_dir = inv_move_map.get(predicted_move, None)
    if move_dir == "UP":
        new_pos = (x, y-1)
    elif move_dir == "DOWN":
        new_pos = (x, y+1)
    elif move_dir == "LEFT":
        new_pos = (x-1, y)
    elif move_dir == "RIGHT":
        new_pos = (x+1, y)
    else:
        new_pos = (x, y)
    if valid_move(new_pos):
        ghost.pos = new_pos

# --- Main program ---

generate_maze_recursive_backtracking(MAZE_WIDTH, MAZE_HEIGHT)

player_start, ghost_start = pick_start_positions(maze)
player = Player(player_start)
ghost = Ghost(ghost_start)

username = get_username()
countdown_timer(3)

# Load model for player (initially None)
model = train_model(username)

# Game variables
game_over = False
move_delay = 150  # ms
last_move_time = pygame.time.get_ticks()
ghost_move_delay = move_delay  # ghost moves every 500 milliseconds (half a second)
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
    if keys[pygame.K_UP]:
        player_move = "UP"
    elif keys[pygame.K_DOWN]:
        player_move = "DOWN"
    elif keys[pygame.K_LEFT]:
        player_move = "LEFT"
    elif keys[pygame.K_RIGHT]:
        player_move = "RIGHT"


    current_time = pygame.time.get_ticks()
    if player_move and current_time - last_move_time > move_delay:
        player.move(player_move)
        last_move_time = current_time

        # Collect training data with label being ghost move direction after player move
        # We can label with ghost's next move (or player's move?), here we just do dummy label for example
        # In a real setup, you'd collect ghost's move as label after the move

        # For now, label ghost's next actual move direction, to be updated later

    if current_time - last_ghost_move_time > ghost_move_delay:
        move_ghost_bfs(ghost, player.pos, maze)
        last_ghost_move_time = current_time

    # Check collision
    if player.pos == ghost.pos:
        game_over = True

    # Draw characters
    player.draw()
    ghost.draw()

    draw_text(f"Player: {username}", (100, HEIGHT - 20), font_game, WHITE)

    pygame.display.flip()
    clock.tick(60)

# Game over screen
screen.fill(BLACK)
draw_text("Game Over!", (WIDTH // 2, HEIGHT // 2 - 50), font_input, RED)
draw_text(f"{username} was caught!", (WIDTH // 2, HEIGHT // 2 + 50), font_input, RED)
pygame.display.flip()
pygame.time.delay(3000)
pygame.quit()
sys.exit()
