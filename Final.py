import pygame
import sys
import csv
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import random
import copy  # For deep-copying the maze layout

# --- Config ---
move_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
inv_move_map = {v: k for k, v in move_map.items()}
reverse_moves = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
ghost_move_delay = 150
data_filename_template = "ghost{}_data_{}.csv"
model_filename_template = "ghost{}_model_{}.joblib"
username_emoji_map = {"bunny": "🐰", "owl": "🦉"}
ghost_frame_delay = 200  # Milliseconds per animation frame for chomping

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "Models")

# Ensure the Models directory exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# --- Maze Definition (Original Layout) ---
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

# --- Pygame Init ---
MAZE_WIDTH = len(maze_layout[0])
MAZE_HEIGHT = len(maze_layout)
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
font_win = pygame.font.SysFont("segoeuiemoji", 60)  # Larger font for winning screen
clock = pygame.time.Clock()

# Color definitions
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 192, 203)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)

# --- Particle Effect for Winning Screen ---
class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = random.randint(3, 8)
        self.color = random.choice([YELLOW, RED, GREEN, CYAN])
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.lifetime = random.randint(30, 60)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1

    def draw(self):
        if self.lifetime > 0:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

# --- Maze Utils ---
def find_tile(maze, tile_char):
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == tile_char:
                return (x, y)
    return (1, 1)

def draw_text(text, pos, font, color=WHITE):
    txt_surface = font.render(text, True, color)
    rect = txt_surface.get_rect(center=pos)
    screen.blit(txt_surface, rect)

def draw_maze(maze):
    for y, row in enumerate(maze):
        for x, tile in enumerate(row):
            rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if tile == '#':
                pygame.draw.rect(screen, BLUE, rect)
            else:
                pygame.draw.rect(screen, BLACK, rect)
            if (x, y) in dots:
                pygame.draw.circle(screen, YELLOW, rect.center, 4)

def valid_move(pos, maze):
    x, y = pos
    return 0 <= y < MAZE_HEIGHT and 0 <= x < MAZE_WIDTH and maze[y][x] != '#'

def move(pos, direction, maze):
    x, y = pos
    if direction == "LEFT": x -= 1
    elif direction == "RIGHT": x += 1
    elif direction == "UP": y -= 1
    elif direction == "DOWN": y += 1
    return (x, y) if valid_move((x, y), maze) else pos

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

def show_winning_screen(username, score):
    particles = []
    flash_timer = 0
    flash_colors = [YELLOW, CYAN, GREEN, RED]
    color_index = 0
    running = True

    while running:
        # Add new particles
        if random.random() < 0.3:  # 30% chance to spawn a particle each frame
            particles.append(Particle(WIDTH // 2, HEIGHT // 2))

        # Update and draw particles
        screen.fill(BLACK)
        for particle in particles[:]:
            particle.update()
            particle.draw()
            if particle.lifetime <= 0:
                particles.remove(particle)

        # Flash the "You Win!" text with different colors
        flash_timer += 1
        if flash_timer > 15:  # Change color every 15 frames
            flash_timer = 0
            color_index = (color_index + 1) % len(flash_colors)

        # Draw winning text
        draw_text("🎉 You Win! 🎉", (WIDTH // 2, HEIGHT // 2 - 100), font_win, flash_colors[color_index])
        draw_text(f"{username} cleared the maze!", (WIDTH // 2, HEIGHT // 2), font_input, WHITE)
        draw_text(f"Final Score: {score}", (WIDTH // 2, HEIGHT // 2 + 80), font_input, YELLOW)
        draw_text("Press R to Restart or Q to Quit", (WIDTH // 2, HEIGHT // 2 + 160), font_game, CYAN)

        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return True  # Restart the game
                if event.key == pygame.K_q:
                    return False  # Quit the game

        clock.tick(60)

class Player:
    def __init__(self, pos, emoji):
        self.pos = pos
        self.last_move = "UP"
        self.emoji = emoji
    def move(self, direction, maze):
        new_pos = move(self.pos, direction, maze)
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
        # Try to load sprite images from the Models directory
        try:
            self.sprites = [
                pygame.image.load(os.path.join(models_dir, f"ghost{id}_closed.png")),
                pygame.image.load(os.path.join(models_dir, f"ghost{id}_open.png"))
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
def predict_player_move(model, ghost_pos, player_pos, other_ghost_pos):
    dx = player_pos[0] - ghost_pos[0]
    dy = player_pos[1] - ghost_pos[1]
    dx_other_ghost = player_pos[0] - other_ghost_pos[0]
    dy_other_ghost = player_pos[1] - other_ghost_pos[1]
    features = pd.DataFrame([[player_pos[0], player_pos[1], ghost_pos[0], ghost_pos[1], dx, dy, dx_other_ghost, dy_other_ghost]],
                            columns=["px", "py", "gx", "gy", "dx", "dy", "dx_other_ghost", "dy_other_ghost"])
    pred = model.predict(features)[0]
    return inv_move_map[pred]

def ghost_ai_move(ghost, player_pos, other_ghost_pos, model, maze):
    if model:
        move_dir = predict_player_move(model, ghost.pos, player_pos, other_ghost_pos)
        new_pos = move(ghost.pos, move_dir, maze)
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
                new_pos = move(ghost.pos, move_dir, maze)
                if new_pos != ghost.pos:
                    dist = manhattan_distance(new_pos, player_pos)
                    if dist < min_distance:
                        min_distance = dist
                        best_move = move_dir
            if best_move:
                ghost.pos = move(ghost.pos, best_move, maze)
                ghost.last_move = best_move
                print(f"Ghost {ghost.id} fallback move: {best_move}, New pos: {ghost.pos}")

# Main Game Function
def run_game():
    global score, dots
    # Reset game state
    score = 0
    # Deep copy the maze layout to ensure a fresh maze each game
    maze = copy.deepcopy([list(row) for row in maze_layout])
    dots = set()
    for y, row in enumerate(maze):
        for x, tile in enumerate(row):
            if tile == '.' or tile == ' ':
                dots.add((x, y))

    # Setup
    player_start = find_tile(maze, "P")
    ghost1_start = find_tile(maze, "G")
    ghost2_start = find_tile(maze, "H")
    # Replace starting positions with dots in the maze
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
    csv_file1 = os.path.join(models_dir, data_filename_template.format(1, username))
    csv_file2 = os.path.join(models_dir, data_filename_template.format(2, username))
    model1_file = os.path.join(models_dir, model_filename_template.format(1, username))
    model2_file = os.path.join(models_dir, model_filename_template.format(2, username))

    feature_names = ["px", "py", "gx", "gy", "dx", "dy", "dx_other_ghost", "dy_other_ghost"]
    class_names = [inv_move_map[i] for i in sorted(inv_move_map.keys())]

    print("\n--- Initializing Ghost AI Models ---")

    # Try loading models from .joblib files first
    if os.path.exists(model1_file):
        try:
            model1 = joblib.load(model1_file)
            print(f"Ghost 1 model loaded from {model1_file}")
        except Exception as e:
            print(f"Error loading Ghost 1 model from {model1_file}: {e}")
    else:
        print(f"Ghost 1 model not found at {model1_file}. Falling back to CSV training.")

    if os.path.exists(model2_file):
        try:
            model2 = joblib.load(model2_file)
            print(f"Ghost 2 model loaded from {model2_file}")
        except Exception as e:
            print(f"Error loading Ghost 2 model from {model2_file}: {e}")
    else:
        print(f"Ghost 2 model not found at {model2_file}. Falling back to CSV training.")

    # If models weren't loaded, train them from CSV files
    if model1 is None:
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
                model1 = RandomForestClassifier(random_state=42).fit(X, y)
                print(f"Ghost 1 model trained with {len(df)} balanced samples from {csv_file1}.")
                # Save the trained model to .joblib
                try:
                    joblib.dump(model1, model1_file)
                    print(f"Ghost 1 model saved to {model1_file}")
                except Exception as e:
                    print(f"Error saving Ghost 1 model to {model1_file}: {e}")
            else:
                print(f"Ghost 1 model not trained: CSV has fewer than 1 entry.")
        else:
            print(f"Ghost 1 model not trained: CSV file '{csv_file1}' not found.")

    if model2 is None:
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
                model2 = RandomForestClassifier(random_state=42).fit(X, y)
                print(f"Ghost 2 model trained with {len(df)} balanced samples from {csv_file2}.")
                # Save the trained model to .joblib
                try:
                    joblib.dump(model2, model2_file)
                    print(f"Ghost 2 model saved to {model2_file}")
                except Exception as e:
                    print(f"Error saving Ghost 2 model to {model2_file}: {e}")
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
        draw_maze(maze)

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
            player.move(player_move, maze)
            last_move_time = current_time
            dx1 = player.pos[0] - ghost1.pos[0]
            dy1 = player.pos[1] - ghost1.pos[1]
            dx2 = player.pos[0] - ghost2.pos[0]
            dy2 = player.pos[1] - ghost2.pos[1]
            dx1_other = player.pos[0] - ghost2.pos[0]  # For ghost1, other ghost is ghost2
            dy1_other = player.pos[1] - ghost2.pos[1]
            dx2_other = player.pos[0] - ghost1.pos[0]  # For ghost2, other ghost is ghost1
            dy2_other = player.pos[1] - ghost1.pos[1]
            log1.append({
                "px": player.pos[0], "py": player.pos[1],
                "gx": ghost1.pos[0], "gy": ghost1.pos[1],
                "dx": dx1, "dy": dy1,
                "dx_other_ghost": dx1_other, "dy_other_ghost": dy1_other,
                "move": move_map[player_move]
            })
            log2.append({
                "px": player.pos[0], "py": player.pos[1],
                "gx": ghost2.pos[0], "gy": ghost2.pos[1],
                "dx": dx2, "dy": dy2,
                "dx_other_ghost": dx2_other, "dy_other_ghost": dy2_other,
                "move": move_map[player_move]
            })

        if current_time - last_ghost_move_time > ghost_move_delay:
            ghost_ai_move(ghost1, player.pos, ghost2.pos, model1, maze)
            ghost_ai_move(ghost2, player.pos, ghost1.pos, model2, maze)
            last_ghost_move_time = current_time

        # Update ghost animations
        ghost1.update_animation()
        ghost2.update_animation()

        if player.pos == ghost1.pos or player.pos == ghost2.pos:
            game_over = True
            return username, False  # Return username and game_over status

        if not dots:
            # Show winning screen and decide whether to restart
            restart = show_winning_screen(username, score)
            if restart:
                return username, True  # Restart the game
            else:
                return username, False  # Quit the game

        if player.pos in dots:
            dots.remove(player.pos)
            score += 10

        player.draw()
        ghost1.draw()
        ghost2.draw()
        draw_text(f"Player: {username}  Score: {score}", (WIDTH // 2, HEIGHT - 20), font_game, WHITE)

        pygame.display.flip()
        clock.tick(60)

    return username, game_over

# Game Over Screen
def show_game_over_screen(username, score):
    screen.fill(BLACK)
    draw_text("Game Over!", (WIDTH // 2, HEIGHT // 2 - 50), font_input, RED)
    draw_text(f"{username} was caught!", (WIDTH // 2, HEIGHT // 2 + 50), font_input, RED)
    draw_text(f"Final Score: {score}", (WIDTH // 2, HEIGHT // 2 + 100), font_input, WHITE)
    draw_text("Press R to Restart or Q to Quit", (WIDTH // 2, HEIGHT // 2 + 160), font_game, CYAN)
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return True  # Restart the game
                if event.key == pygame.K_q:
                    return False  # Quit the game
        clock.tick(30)

# Main Program Loop
running = True
username = None
while running:
    if username is None:
        username_result, restart = run_game()
        username = username_result
    else:
        # Run the game with the same username
        _, restart = run_game()

    if not restart:
        # Show game over screen if the player lost
        restart = show_game_over_screen(username, score)

    # Save movement data for both ghosts
    log1 = []  # Reset logs for new game
    log2 = []
    if log1:
        df1 = pd.DataFrame(log1)
        csv_file1 = os.path.join(models_dir, data_filename_template.format(1, username))
        if os.path.exists(csv_file1):
            old_df = pd.read_csv(csv_file1)
            df1 = pd.concat([old_df, df1], ignore_index=True)
        try:
            df1.to_csv(csv_file1, index=False)
            print(f"Ghost 1 data saved to {csv_file1}")
        except Exception as e:
            print(f"Error saving Ghost 1 data to {csv_file1}: {e}")

    if log2:
        df2 = pd.DataFrame(log2)
        csv_file2 = os.path.join(models_dir, data_filename_template.format(2, username))
        if os.path.exists(csv_file2):
            old_df = pd.read_csv(csv_file2)
            df2 = pd.concat([old_df, df2], ignore_index=True)
        try:
            df2.to_csv(csv_file2, index=False)
            print(f"Ghost 2 data saved to {csv_file2}")
        except Exception as e:
            print(f"Error saving Ghost 2 data to {csv_file2}: {e}")

    # Retrain and save models with updated data
    if log1 and os.path.exists(csv_file1):
        df = pd.read_csv(csv_file1)
        if len(df) >= 1:
            move_counts = df["move"].value_counts()
            min_count = move_counts.min()
            df = pd.concat([df[df["move"] == move].sample(min_count) for move in move_counts.index])
            df = df.sample(n=min(40000, len(df)), random_state=42)
            X = df[["px", "py", "gx", "gy", "dx", "dy", "dx_other_ghost", "dy_other_ghost"]]
            y = df["move"]
            model1 = RandomForestClassifier(random_state=42).fit(X, y)
            model1_file = os.path.join(models_dir, model_filename_template.format(1, username))
            try:
                joblib.dump(model1, model1_file)
                print(f"Updated Ghost 1 model saved to {model1_file}")
            except Exception as e:
                print(f"Error saving updated Ghost 1 model to {model1_file}: {e}")

    if log2 and os.path.exists(csv_file2):
        df = pd.read_csv(csv_file2)
        if len(df) >= 1:
            move_counts = df["move"].value_counts()
            min_count = move_counts.min()
            df = pd.concat([df[df["move"] == move].sample(min_count) for move in move_counts.index])
            df = df.sample(n=min(40000, len(df)), random_state=42)
            X = df[["px", "py", "gx", "gy", "dx", "dy", "dx_other_ghost", "dy_other_ghost"]]
            y = df["move"]
            model2 = RandomForestClassifier(random_state=42).fit(X, y)
            model2_file = os.path.join(models_dir, model_filename_template.format(2, username))
            try:
                joblib.dump(model2, model2_file)
                print(f"Updated Ghost 2 model saved to {model2_file}")
            except Exception as e:
                print(f"Error saving updated Ghost 2 model to {model2_file}: {e}")

    if not restart:
        running = False

pygame.quit()
sys.exit()