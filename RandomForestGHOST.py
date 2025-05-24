import pygame
import sys
import csv
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

# --- Config ---
move_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
inv_move_map = {v: k for k, v in move_map.items()}
reverse_moves = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
ghost_move_delay = 150
data_filename_template = "ghost{}_data_{}.csv"
username_emoji_map = {"bunny": "🐰", "owl": "🦉"}
ghost_frame_delay = 200  # Milliseconds per animation frame for chomping

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
    "#...##...........H....##...#",
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
pygame.display.set_caption("Pac-Man with Two Adaptive Ghosts")
font_input = pygame.font.SysFont("segoeuiemoji", 48)
font_countdown = pygame.font.SysFont("segoeuiemoji", 200)
font_game = pygame.font.SysFont("segoeuiemoji", 36)
font_player = pygame.font.SysFont("segoeuiemoji", 36)
clock = pygame.time.Clock()

# Color definitions
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 192, 203)
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

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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
    username = username.lower()
    emoji = username_emoji_map.get(username, "")
    return username, emoji

def countdown_timer(seconds=3):
    for i in range(seconds, 0, -1):
        screen.fill(BLACK)
        draw_text(str(i), (WIDTH // 2, HEIGHT // 2), font_countdown)
        pygame.display.flip()
        pygame.time.delay(1000)

class Player:
    def __init__(self, pos, emoji):
        self.pos = pos
        self.last_move = "UP"
        self.emoji = emoji
    def move(self, direction):
        new_pos = move(self.pos, direction)
        if new_pos != self.pos:
            self.pos = new_pos
            self.last_move = direction
    def draw(self):
        rect = pygame.Rect(self.pos[0]*TILE_SIZE, self.pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        draw_text(self.emoji, rect.center, font_player, YELLOW)

class Ghost:
    def __init__(self, pos, color, id):
        self.pos = pos
        self.color = color
        self.id = id
        self.last_move = None
        self.frame = 0  # 0 for closed mouth, 1 for open mouth
        self.last_frame_time = pygame.time.get_ticks()
        # Try to load sprite images
        try:
            self.sprites = [
                pygame.image.load(f"ghost{id}_closed.png"),
                pygame.image.load(f"ghost{id}_open.png")
            ]
            self.sprites = [pygame.transform.scale(sprite, (TILE_SIZE, TILE_SIZE)) for sprite in self.sprites]
            self.use_images = True
        except FileNotFoundError:
            self.use_images = False  # Fallback to programmatic drawing
    def draw(self):
        rect = pygame.Rect(self.pos[0]*TILE_SIZE, self.pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        if self.use_images:
            screen.blit(self.sprites[self.frame], rect.topleft)
        else:
            # Fallback: Draw circle with chomping mouth
            pygame.draw.circle(screen, self.color, rect.center, TILE_SIZE//2 - 4)
            if self.frame == 1:  # Open mouth
                mouth_points = [
                    (rect.center[0], rect.center[1] - 5),
                    (rect.center[0] + 10, rect.center[1] + 5),
                    (rect.center[0] - 10, rect.center[1] + 5)
                ]
                pygame.draw.polygon(screen, WHITE, mouth_points)
    def update_animation(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_frame_time > ghost_frame_delay:
            self.frame = (self.frame + 1) % 2  # Switch between 0 and 1
            self.last_frame_time = current_time

# Ghost AI movement using prediction
def predict_player_move(model, ghost_pos, player_pos):
    dx = player_pos[0] - ghost_pos[0]
    dy = player_pos[1] - ghost_pos[1]
    features = pd.DataFrame([[player_pos[0], player_pos[1], ghost_pos[0], ghost_pos[1], dx, dy]],
                            columns=["px", "py", "gx", "gy", "dx", "dy"])
    pred = model.predict(features)[0]
    return inv_move_map[pred]

def ghost_ai_move(ghost, player_pos, model):
    if model:
        move_dir = predict_player_move(model, ghost.pos, player_pos)
        new_pos = move(ghost.pos, move_dir)
        current_distance = manhattan_distance(ghost.pos, player_pos)
        reverse_move = reverse_moves.get(ghost.last_move) if ghost.last_move else None
        print(f"Ghost {ghost.id} predicted move: {move_dir}, Current pos: {ghost.pos}, Attempted pos: {new_pos}, Last move: {ghost.last_move}, Excluded reverse: {reverse_move}")
        if new_pos != ghost.pos and manhattan_distance(new_pos, player_pos) <= current_distance and move_dir != reverse_move:
            ghost.pos = new_pos
            ghost.last_move = move_dir
        else:
            possible_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
            if reverse_move:
                possible_moves.remove(reverse_move)
            np.random.shuffle(possible_moves)
            best_move = None
            min_distance = float('inf')
            for move_dir in possible_moves:
                new_pos = move(ghost.pos, move_dir)
                if new_pos != ghost.pos:
                    dist = manhattan_distance(new_pos, player_pos)
                    if dist < min_distance:
                        min_distance = dist
                        best_move = move_dir
            if best_move:
                ghost.pos = move(ghost.pos, best_move)
                ghost.last_move = best_move
                print(f"Ghost {ghost.id} fallback move: {best_move}, New pos: {ghost.pos}")

# Setup
player_start = find_tile("P")
ghost1_start = find_tile("G")
ghost2_start = find_tile("H")
maze[player_start[1]][player_start[0]] = '.'
maze[ghost1_start[1]][ghost1_start[0]] = '.'
maze[ghost2_start[1]][ghost2_start[0]] = '.'
username, player_emoji = get_username()
player = Player(player_start, player_emoji)
ghost1 = Ghost(ghost1_start, RED, 1)
ghost2 = Ghost(ghost2_start, PINK, 2)
countdown_timer()

# Load models for both ghosts
model1 = None
model2 = None
log1 = []
log2 = []
csv_file1 = data_filename_template.format(1, username)
csv_file2 = data_filename_template.format(2, username)

feature_names = ["px", "py", "gx", "gy", "dx", "dy"]
class_names = [inv_move_map[i] for i in sorted(inv_move_map.keys())]

print("\n--- Initializing Ghost AI Models ---")

if os.path.exists(csv_file1):
    df = pd.read_csv(csv_file1)
    if len(df) >= 1:
        move_counts = df["move"].value_counts()
        print(f"Ghost 1 move distribution: {move_counts.to_dict()}")
        min_count = move_counts.min()
        df = pd.concat([df[df["move"] == move].sample(min_count) for move in move_counts.index])
        df = df.sample(n=min(40000, len(df)), random_state=42)
        X = df[feature_names]
        y = df["move"]
        model1 = DecisionTreeClassifier(random_state=42).fit(X, y)
        print(f"Ghost 1 model trained with {len(df)} balanced samples from {csv_file1}.")
        print("Decision Tree for Ghost 1:")
        print(export_text(model1, feature_names=feature_names, class_names=class_names))
    else:
        print(f"Ghost 1 model not trained: CSV has fewer than 1 entry.")
else:
    print(f"Ghost 1 model not trained: CSV file '{csv_file1}' not found.")

if os.path.exists(csv_file2):
    df = pd.read_csv(csv_file2)
    if len(df) >= 1:
        move_counts = df["move"].value_counts()
        print(f"Ghost 2 move distribution: {move_counts.to_dict()}")
        min_count = move_counts.min()
        df = pd.concat([df[df["move"] == move].sample(min_count) for move in move_counts.index])
        df = df.sample(n=min(40000, len(df)), random_state=42)
        X = df[feature_names]
        y = df["move"]
        model2 = DecisionTreeClassifier(random_state=42).fit(X, y)
        print(f"Ghost 2 model trained with {len(df)} balanced samples from {csv_file2}.")
        print("Decision Tree for Ghost 2:")
        print(export_text(model2, feature_names=feature_names, class_names=class_names))
    else:
        print(f"Ghost 2 model not trained: CSV has fewer than 1 entry.")
else:
    print(f"Ghost 2 model not trained: CSV file '{csv_file2}' not found.")

print("-----------------------------------")

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
        dx1 = player.pos[0] - ghost1.pos[0]
        dy1 = player.pos[1] - ghost1.pos[1]
        dx2 = player.pos[0] - ghost2.pos[0]
        dy2 = player.pos[1] - ghost2.pos[1]
        log1.append({
            "px": player.pos[0], "py": player.pos[1],
            "gx": ghost1.pos[0], "gy": ghost1.pos[1],
            "dx": dx1, "dy": dy1,
            "move": move_map[player_move]
        })
        log2.append({
            "px": player.pos[0], "py": player.pos[1],
            "gx": ghost2.pos[0], "gy": ghost2.pos[1],
            "dx": dx2, "dy": dy2,
            "move": move_map[player_move]
        })

    if current_time - last_ghost_move_time > ghost_move_delay:
        ghost_ai_move(ghost1, player.pos, model1)
        ghost_ai_move(ghost2, player.pos, model2)
        last_ghost_move_time = current_time

    # Update ghost animations
    ghost1.update_animation()
    ghost2.update_animation()

    if player.pos == ghost1.pos or player.pos == ghost2.pos:
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
    ghost1.draw()
    ghost2.draw()
    draw_text(f"Player: {username}  Score: {score}", (WIDTH // 2, HEIGHT - 20), font_game, WHITE)

    pygame.display.flip()
    clock.tick(60)

# Save movement data for both ghosts
if log1:
    df1 = pd.DataFrame(log1)
    if os.path.exists(csv_file1):
        old_df = pd.read_csv(csv_file1)
        df1 = pd.concat([old_df, df1], ignore_index=True)
    df1.to_csv(csv_file1, index=False)

if log2:
    df2 = pd.DataFrame(log2)
    if os.path.exists(csv_file2):
        old_df = pd.read_csv(csv_file2)
        df2 = pd.concat([old_df, df2], ignore_index=True)
    df2.to_csv(csv_file2, index=False)

# End screen
screen.fill(BLACK)
draw_text("Game Over!", (WIDTH // 2, HEIGHT // 2 - 50), font_input, RED)
draw_text(f"{username} was caught!", (WIDTH // 2, HEIGHT // 2 + 50), font_input, RED)
draw_text(f"Final Score: {score}", (WIDTH // 2, HEIGHT // 2 + 100), font_input, WHITE)
pygame.display.flip()
pygame.time.delay(3000)
pygame.quit()
sys.exit()