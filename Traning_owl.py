import pygame
import sys
import csv
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
import numpy as np
from heapq import heappush, heappop
import joblib

# --- Config ---
move_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
inv_move_map = {v: k for k, v in move_map.items()}
ghost_move_delay = 250
player_move_delay = 150
data_filename_template = "ghost{}_data_{}.csv"
model_filename_template = "ghost{}_model_{}.joblib"

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
    "######.#####.##.#####.######",
    "######.#####.##.#####.######",
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
    "#.......G........G2........#",
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
pygame.display.set_caption("Pac-Man Training with Two Adaptive Ghosts")
font_input = pygame.font.SysFont(None, 48)
font_countdown = pygame.font.SysFont(None, 200)
font_game = pygame.font.SysFont(None, 36)
font_ghost_info = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

# Color definitions
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 192, 203)
BLUE = (0, 0, 255)

# --- Maze Utils ---
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

def draw_maze(dots_set):
    for y, row in enumerate(maze):
        for x, tile in enumerate(row):
            rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if tile == '#':
                pygame.draw.rect(screen, BLUE, rect)
            else:
                pygame.draw.rect(screen, BLACK, rect)
            if (x, y) in dots_set:
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

# --- A* Pathfinding for AI Player ---
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def a_star(start, goal, positions_to_avoid=set()):
    open_set = [(0, start, [])]
    closed_set = set()
    g_score = {start: 0}

    while open_set:
        current_f_overall, current, path = heappop(open_set)
        
        if current == goal:
            return path[0] if path else None

        if current in closed_set:
            continue
        closed_set.add(current)

        x, y = current
        neighbors_defs = [(x+1, y, "RIGHT"), (x-1, y, "LEFT"), (x, y+1, "DOWN"), (x, y-1, "UP")]

        for nx, ny, direction in neighbors_defs:
            neighbor = (nx, ny)
            if not valid_move(neighbor) or neighbor in closed_set or neighbor in positions_to_avoid:
                continue

            tentative_g = g_score[current] + 1
            
            if neighbor in g_score and tentative_g >= g_score[neighbor]:
                continue

            g_score[neighbor] = tentative_g
            h_score = manhattan_distance(neighbor, goal)
            f_score_neighbor = tentative_g + h_score
            heappush(open_set, (f_score_neighbor, neighbor, path + [direction]))
            
    return None

# --- Player and Ghost Classes ---
class Player:
    def __init__(self, pos):
        self.pos = pos
        self.last_move = "UP"
    def move(self, direction_str):
        new_pos = move(self.pos, direction_str)
        if new_pos != self.pos:
            self.pos = new_pos
            self.last_move = direction_str
    def draw(self):
        rect = pygame.Rect(self.pos[0]*TILE_SIZE, self.pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.circle(screen, YELLOW, rect.center, TILE_SIZE//2 - 4)

class Ghost:
    def __init__(self, pos, color, id_num):
        self.pos = pos
        self.color = color
        self.id = id_num
        self.current_strategy = "Idle"
    def draw(self):
        rect = pygame.Rect(self.pos[0]*TILE_SIZE, self.pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.circle(screen, self.color, rect.center, TILE_SIZE//2 - 4)

# --- Ghost AI ---
def predict_player_move(model, ghost_pos, player_pos, other_ghost_pos):
    dx = player_pos[0] - ghost_pos[0]
    dy = player_pos[1] - ghost_pos[1]
    dx_other_ghost = player_pos[0] - other_ghost_pos[0]
    dy_other_ghost = player_pos[1] - other_ghost_pos[1]

    features = pd.DataFrame([[
        player_pos[0], player_pos[1], 
        ghost_pos[0], ghost_pos[1], 
        dx, dy, 
        dx_other_ghost, dy_other_ghost
    ]], columns=["px", "py", "gx", "gy", "dx", "dy", "dx_other_ghost", "dy_other_ghost"])
    
    try:
        pred_numeric = model.predict(features)[0]
        return inv_move_map[pred_numeric]
    except Exception as e:
        return "UP"

def ghost_ai_move(ghost, player_pos, other_ghost_pos, model):
    next_move_direction = None
    positions_to_avoid_for_ghost = {other_ghost_pos} if other_ghost_pos else set()
    
    if model:
        predicted_player_direction = predict_player_move(model, ghost.pos, player_pos, other_ghost_pos)
        predicted_player_next_pos = move(player_pos, predicted_player_direction)
        next_move_direction = a_star(ghost.pos, predicted_player_next_pos, positions_to_avoid_for_ghost)

        if next_move_direction:
            ghost.current_strategy = "Model-Intercept"
        else:
            ghost.current_strategy = "A*-Chase"
            next_move_direction = a_star(ghost.pos, player_pos, positions_to_avoid_for_ghost)
    
    if not next_move_direction:
        if model:
            ghost.current_strategy = "A*-Chase (Fallback)"
            next_move_direction = a_star(ghost.pos, player_pos, positions_to_avoid_for_ghost)
        else:
            ghost.current_strategy = "A*-Chase (No Model)"
            next_move_direction = a_star(ghost.pos, player.pos, positions_to_avoid_for_ghost)

    if not next_move_direction:
        ghost.current_strategy = "Random"
        possible_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
        np.random.shuffle(possible_moves)
        for move_dir in possible_moves:
            new_pos = move(ghost.pos, move_dir)
            if new_pos != ghost.pos and new_pos not in positions_to_avoid_for_ghost:
                next_move_direction = move_dir
                break
    
    if next_move_direction:
        ghost.pos = move(ghost.pos, next_move_direction)

# --- AI Player Logic ---
def ai_player_move(player, current_dots, ghost1, ghost2, username):
    if not current_dots:
        return None

    target_dot = None

    if username == "bunny":
        # Bunny's strategy: Prioritize nearest dot, adjusted for ghost proximity
        min_score = float('inf')
        for dot in current_dots:
            dist_to_dot = manhattan_distance(player.pos, dot)
            dist_dot_to_ghost1 = manhattan_distance(dot, ghost1.pos)
            dist_dot_to_ghost2 = manhattan_distance(dot, ghost2.pos)
            
            risk_score = dist_to_dot
            if dist_dot_to_ghost1 < 5:
                risk_score += (5 - dist_dot_to_ghost1) * 20
            if dist_dot_to_ghost2 < 5:
                risk_score += (5 - dist_dot_to_ghost2) * 20

            if risk_score < min_score:
                min_score = risk_score
                target_dot = dot

    elif username == "owl":
        # Owl's strategy: Prioritize dots farthest from ghosts, even if farther from player
        max_safety_score = float('-inf')
        for dot in current_dots:
            dist_to_dot = manhattan_distance(player.pos, dot)
            dist_dot_to_ghost1 = manhattan_distance(dot, ghost1.pos)
            dist_dot_to_ghost2 = manhattan_distance(dot, ghost2.pos)
            
            # Safety score: Favor dots farther from ghosts, penalize distance from player slightly
            safety_score = min(dist_dot_to_ghost1, dist_dot_to_ghost2) - (dist_to_dot * 0.5)
            if dist_dot_to_ghost1 < 3 or dist_dot_to_ghost2 < 3:
                safety_score -= 100  # Heavier penalty for very close ghosts

            if safety_score > max_safety_score:
                max_safety_score = safety_score
                target_dot = dot

    if not target_dot:
        target_dot = list(current_dots)[0]

    ghost_positions_set = {ghost1.pos, ghost2.pos}
    direction_to_move = a_star(player.pos, target_dot, ghost_positions_set)
    
    # Add randomness for owl to further differentiate behavior
    if username == "owl" and np.random.random() < 0.2:  # 20% chance for owl to move randomly
        possible_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
        np.random.shuffle(possible_moves)
        for move_dir in possible_moves:
            new_pos = move(player.pos, move_dir)
            if new_pos != player.pos and new_pos not in ghost_positions_set:
                return move_dir
    
    if not direction_to_move:
        possible_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
        np.random.shuffle(possible_moves)
        for move_dir in possible_moves:
            new_pos = move(player.pos, move_dir)
            if new_pos != player.pos and new_pos not in ghost_positions_set:
                return move_dir
        return None
        
    return direction_to_move

# --- Main Game Reset Function ---
def reset_game():
    global score, dots, player, ghost1, ghost2, maze
    maze[:] = [list(row) for row in maze_layout]
    score = 0
    dots = set()
    for y_idx, row_str in enumerate(maze_layout):
        for x_idx, tile_char in enumerate(row_str):
            if tile_char == '.':
                dots.add((x_idx, y_idx))
    player_start_pos = find_tile("P")
    ghost1_start_pos = find_tile("G")
    ghost2_start_pos = find_tile("G2")
    player = Player(player_start_pos)
    ghost1 = Ghost(ghost1_start_pos, RED, 1)
    ghost2 = Ghost(ghost2_start_pos, PINK, 2)
    if player_start_pos != (1, 1):
        maze[player_start_pos[1]][player_start_pos[0]] = '.'
    if ghost1_start_pos != (1, 1):
        maze[ghost1_start_pos[1]][ghost1_start_pos[0]] = '.'
    if ghost2_start_pos != (1, 1):
        maze[ghost2_start_pos[1]][ghost2_start_pos[0]] = '.'

# --- Main Training Loop ---
username = get_username()
game_count = 0
iteration = 0
player = None
ghost1 = None
ghost2 = None
score = 0
dots = set()

model1 = None
model2 = None

feature_names = ["px", "py", "gx", "gy", "dx", "dy", "dx_other_ghost", "dy_other_ghost"]
class_names = [inv_move_map[i] for i in sorted(inv_move_map.keys())]

while iteration < 500:
    game_count += 1
    iteration += 1
    print(f"\n--- Starting Game {game_count} (Overall Iteration {iteration}) ---")
    reset_game()
    countdown_timer()

    log1 = []
    log2 = []
    csv_file1 = data_filename_template.format(1, username)
    csv_file2 = data_filename_template.format(2, username)
    model_file1 = model_filename_template.format(1, username)
    model_file2 = model_filename_template.format(2, username)

    if os.path.exists(model_file1):
        try:
            model1 = joblib.load(model_file1)
            print(f"Loaded existing Ghost 1 model from {model_file1}.")
        except Exception as e:
            print(f"Error loading Ghost 1 model from {model_file1}: {e}. Will attempt to train from CSV.")
            model1 = None
    
    if model1 is None and os.path.exists(csv_file1):
        try:
            df = pd.read_csv(csv_file1)
            if len(df) >= 1:
                if all(col in df.columns for col in feature_names + ["move"]):
                    X = df[feature_names]
                    y = df["move"]
                    if not X.empty and not y.empty:
                        model1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X, y)
                        print(f"Ghost 1 model trained using RandomForestClassifier from {csv_file1}. Total samples: {len(X)}")
                        print("Note: Random Forest consists of many trees, printing the full structure is not practical.")
                    else:
                        print(f"Warning: CSV {csv_file1} is empty after feature selection, cannot train Ghost 1 model.")
                else:
                    print(f"Warning: CSV {csv_file1} missing required columns for training (expected: {feature_names + ['move']}).")
            else:
                print(f"Warning: CSV {csv_file1} has fewer than 1 entry, cannot train Ghost 1 model. Using no model.")
        except pd.errors.EmptyDataError:
            print(f"Warning: CSV {csv_file1} is empty, cannot train Ghost 1 model. Using no model.")
        except Exception as e:
            print(f"Error loading or training Ghost 1 model from CSV: {e}. Using no model.")
    elif model1 is None:
        print(f"Ghost 1 model not trained: CSV file '{csv_file1}' not found. Using no model.")

    if os.path.exists(model_file2):
        try:
            model2 = joblib.load(model_file2)
            print(f"Loaded existing Ghost 2 model from {model_file2}.")
        except Exception as e:
            print(f"Error loading Ghost 2 model from {model_file2}: {e}. Will attempt to train from CSV.")
            model2 = None

    if model2 is None and os.path.exists(csv_file2):
        try:
            df = pd.read_csv(csv_file2)
            if len(df) >= 1:
                if all(col in df.columns for col in feature_names + ["move"]):
                    X = df[feature_names]
                    y = df["move"]
                    if not X.empty and not y.empty:
                        model2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X, y)
                        print(f"Ghost 2 model trained using RandomForestClassifier from {csv_file2}. Total samples: {len(X)}")
                        print("Note: Random Forest consists of many trees, printing the full structure is not practical.")
                    else:
                        print(f"Warning: CSV {csv_file2} is empty after feature selection, cannot train Ghost 2 model.")
                else:
                    print(f"Warning: CSV {csv_file2} missing required columns for training (expected: {feature_names + ['move']}).")
            else:
                print(f"Warning: CSV {csv_file2} has fewer than 1 entry, cannot train Ghost 2 model. Using no model.")
        except pd.errors.EmptyDataError:
            print(f"Warning: CSV {csv_file2} is empty, cannot train Ghost 2 model. Using no model.")
        except Exception as e:
            print(f"Error loading or training Ghost 2 model from CSV: {e}. Using no model.")
    elif model2 is None:
        print(f"Ghost 2 model not trained: CSV file '{csv_file2}' not found. Using no model.")

    print("--------------------------------------------------")

    game_over = False
    last_player_move_time = pygame.time.get_ticks()
    last_ghost_move_time = pygame.time.get_ticks()

    while not game_over:
        screen.fill(BLACK)
        draw_maze(dots)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                if log1:
                    df1_new = pd.DataFrame(log1)
                    if os.path.exists(csv_file1):
                        try:
                            old_df1 = pd.read_csv(csv_file1)
                            df1_combined = pd.concat([old_df1, df1_new], ignore_index=True)
                        except pd.errors.EmptyDataError:
                            df1_combined = df1_new
                    else:
                        df1_combined = df1_new
                    df1_combined.to_csv(csv_file1, index=False)
                if log2:
                    df2_new = pd.DataFrame(log2)
                    if os.path.exists(csv_file2):
                        try:
                            old_df2 = pd.read_csv(csv_file2)
                            df2_combined = pd.concat([old_df2, df2_new], ignore_index=True)
                        except pd.errors.EmptyDataError:
                            df2_combined = df2_new
                    else:
                        df2_combined = df2_new
                    df2_combined.to_csv(csv_file2, index=False)
                pygame.quit()
                sys.exit()

        current_time = pygame.time.get_ticks()

        if current_time - last_player_move_time > player_move_delay:
            player_action_str = ai_player_move(player, dots, ghost1, ghost2, username)
            if player_action_str:
                dx1 = player.pos[0] - ghost1.pos[0]
                dy1 = player.pos[1] - ghost1.pos[1]
                dx_other_ghost_for_g1 = player.pos[0] - ghost2.pos[0]
                dy_other_ghost_for_g1 = player.pos[1] - ghost2.pos[1]

                log1.append({
                    "px": player.pos[0], "py": player.pos[1],
                    "gx": ghost1.pos[0], "gy": ghost1.pos[1],
                    "dx": dx1, "dy": dy1,
                    "dx_other_ghost": dx_other_ghost_for_g1,
                    "dy_other_ghost": dy_other_ghost_for_g1,
                    "move": move_map[player_action_str]
                })

                dx2 = player.pos[0] - ghost2.pos[0]
                dy2 = player.pos[1] - ghost2.pos[1]
                dx_other_ghost_for_g2 = player.pos[0] - ghost1.pos[0]
                dy_other_ghost_for_g2 = player.pos[1] - ghost1.pos[1]

                log2.append({
                    "px": player.pos[0], "py": player.pos[1],
                    "gx": ghost2.pos[0], "gy": ghost2.pos[1],
                    "dx": dx2, "dy": dy2,
                    "dx_other_ghost": dx_other_ghost_for_g2,
                    "dy_other_ghost": dy_other_ghost_for_g2,
                    "move": move_map[player_action_str]
                })
                
                player.move(player_action_str)
            last_player_move_time = current_time

        if current_time - last_ghost_move_time > ghost_move_delay:
            ghost_ai_move(ghost1, player.pos, ghost2.pos, model1)
            ghost_ai_move(ghost2, player.pos, ghost1.pos, model2)
            last_ghost_move_time = current_time

        if player.pos == ghost1.pos or player.pos == ghost2.pos:
            game_over = True
            print("AI caught by a ghost!")

        if player.pos in dots:
            dots.remove(player.pos)
            score += 10

        if not dots:
            screen.fill(BLACK)
            draw_text("AI Wins!", (WIDTH // 2, HEIGHT // 2 - 50), font_input, YELLOW)
            draw_text(f"Score: {score}", (WIDTH // 2, HEIGHT // 2 + 50), font_input, YELLOW)
            pygame.display.flip()
            pygame.time.delay(2000)
            game_over = True
            print("AI wins! All dots collected.")

        player.draw()
        ghost1.draw()
        ghost2.draw()
        
        draw_text(f"G1: {ghost1.current_strategy}", (WIDTH - 120, TILE_SIZE * 2 - 10), font_ghost_info, RED)
        draw_text(f"G2: {ghost2.current_strategy}", (WIDTH - 120, TILE_SIZE * 2 + 10), font_ghost_info, PINK)

        draw_text(f"Player: AI ({username})  Score: {score}  Game: {game_count} Iter: {iteration}",
                    (WIDTH // 2, TILE_SIZE // 2), font_game, WHITE)

        pygame.display.flip()
        clock.tick(60)

    if log1:
        df1_new = pd.DataFrame(log1)
        if os.path.exists(csv_file1):
            try:
                old_df1 = pd.read_csv(csv_file1)
                df1_combined = pd.concat([old_df1, df1_new], ignore_index=True)
            except pd.errors.EmptyDataError:
                df1_combined = df1_new
        else:
            df1_combined = df1_new
        df1_combined.to_csv(csv_file1, index=False)
        print(f"Saved {len(df1_new)} new entries to {csv_file1}. Total entries: {len(df1_combined)}")

    if log2:
        df2_new = pd.DataFrame(log2)
        if os.path.exists(csv_file2):
            try:
                old_df2 = pd.read_csv(csv_file2)
                df2_combined = pd.concat([old_df2, df2_new], ignore_index=True)
            except pd.errors.EmptyDataError:
                df2_combined = df2_new
        else:
            df2_combined = df2_new
        df2_combined.to_csv(csv_file2, index=False)
        print(f"Saved {len(df2_new)} new entries to {csv_file2}. Total entries: {len(df2_combined)}")

    if not dots and game_over:
        pass
    elif game_over:
        screen.fill(BLACK)
        draw_text("Game Over!", (WIDTH // 2, HEIGHT // 2 - 100), font_input, RED)
        if player.pos == ghost1.pos or player.pos == ghost2.pos :
            draw_text("AI was caught!", (WIDTH // 2, HEIGHT // 2 - 50), font_input, RED)
        draw_text(f"Final Score: {score}", (WIDTH // 2, HEIGHT // 2 + 50), font_input, WHITE)
        pygame.display.flip()
        pygame.time.delay(2000)

if model1:
    model_filename1 = model_filename_template.format(1, username)
    joblib.dump(model1, model_filename1)
    print(f"Ghost 1 model saved to {model_filename1}")

if model2:
    model_filename2 = model_filename_template.format(2, username)
    joblib.dump(model2, model_filename2)
    print(f"Ghost 2 model saved to {model_filename2}")

pygame.quit()
sys.exit()