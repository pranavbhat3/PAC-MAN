import pygame
import sys
import csv
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier # Changed from DecisionTreeClassifier
from sklearn.tree import export_text # Kept for potential insight, though less useful for RF directly
import numpy as np
from heapq import heappush, heappop
import joblib # Import joblib for saving models

# --- Config ---
move_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
inv_move_map = {v: k for k, v in move_map.items()}
ghost_move_delay = 250
player_move_delay = 150
data_filename_template = "ghost{}_data_{}.csv"
model_filename_template = "ghost{}_model_{}.joblib" # Template for saving models

# --- Maze Definition ---
# This is the master layout and should not be modified after definition.
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
    "######.##          ##.######", # Corrected extra space and added ' ' for path
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
    "#.......G........G2........#", # Changed 'H' to 'G2' for clarity
    "############################"
]
# This 'maze' variable will be the working copy for each game instance.
# It's initialized here and reset in reset_game().
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
font_ghost_info = pygame.font.SysFont(None, 24) # Smaller font for ghost info
clock = pygame.time.Clock()

# Color definitions
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 192, 203) # Pink for Ghost 2
BLUE = (0, 0, 255)

# --- Maze Utils ---
def find_tile(tile_char):
    # Searches the global 'maze' variable
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == tile_char:
                return (x, y)
    return (1, 1) # Default if not found (e.g., top-left accessible corner)

def draw_text(text, pos, font, color=WHITE):
    txt_surface = font.render(text, True, color)
    rect = txt_surface.get_rect(center=pos)
    screen.blit(txt_surface, rect)

def draw_maze(dots_set): # Renamed parameter to avoid confusion with global 'dots'
    for y, row in enumerate(maze): # Draws based on the current state of 'maze'
        for x, tile in enumerate(row):
            rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            if tile == '#':
                pygame.draw.rect(screen, BLUE, rect)
            else:
                pygame.draw.rect(screen, BLACK, rect)
            if (x, y) in dots_set: # Use the passed 'dots_set'
                pygame.draw.circle(screen, YELLOW, rect.center, 4)

def valid_move(pos):
    x, y = pos
    # Checks against the current state of 'maze'
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
    """
    Finds the first move direction using A* from start to goal,
    avoiding specified positions.
    """
    open_set = [(0, start, [])] # (f_score, position, path)
    closed_set = set()
    g_score = {start: 0}

    while open_set:
        current_f_overall, current, path = heappop(open_set)
        
        if current == goal:
            return path[0] if path else None # Return first move direction string

        if current in closed_set:
            continue
        closed_set.add(current)

        x, y = current
        neighbors_defs = [(x+1, y, "RIGHT"), (x-1, y, "LEFT"), (x, y+1, "DOWN"), (x, y-1, "UP")]

        for nx, ny, direction in neighbors_defs:
            neighbor = (nx, ny)
            # Check for validity and if neighbor is in positions to avoid
            if not valid_move(neighbor) or neighbor in closed_set or neighbor in positions_to_avoid:
                continue

            tentative_g = g_score[current] + 1
            
            if neighbor in g_score and tentative_g >= g_score[neighbor]:
                continue # This path is not better

            g_score[neighbor] = tentative_g
            h_score = manhattan_distance(neighbor, goal)
            f_score_neighbor = tentative_g + h_score
            heappush(open_set, (f_score_neighbor, neighbor, path + [direction]))
            
    return None # No path found

# --- Player and Ghost Classes ---
class Player:
    def __init__(self, pos):
        self.pos = pos
        self.last_move = "UP" # Default last move
    def move(self, direction_str): # Ensure direction is a string like "UP"
        new_pos = move(self.pos, direction_str)
        if new_pos != self.pos:
            self.pos = new_pos
            self.last_move = direction_str # Update last_move with the string
    def draw(self):
        rect = pygame.Rect(self.pos[0]*TILE_SIZE, self.pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.circle(screen, YELLOW, rect.center, TILE_SIZE//2 - 4)

class Ghost:
    def __init__(self, pos, color, id_num): # Changed id to id_num to avoid conflict with builtin
        self.pos = pos
        self.color = color
        self.id = id_num # Store id_num
        self.current_strategy = "Idle" # New attribute to store current strategy
    def draw(self):
        rect = pygame.Rect(self.pos[0]*TILE_SIZE, self.pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.circle(screen, self.color, rect.center, TILE_SIZE//2 - 4)

# --- Ghost AI ---
def predict_player_move(model, ghost_pos, player_pos, other_ghost_pos):
    """
    Predicts the player's next move based on the model and current state.
    Includes other ghost's relative position as a feature.
    """
    dx = player_pos[0] - ghost_pos[0]
    dy = player_pos[1] - ghost_pos[1]
    
    # Calculate relative position to the OTHER ghost as well
    dx_other_ghost = player_pos[0] - other_ghost_pos[0]
    dy_other_ghost = player_pos[1] - other_ghost_pos[1]

    features = pd.DataFrame([[
        player_pos[0], player_pos[1], 
        ghost_pos[0], ghost_pos[1], 
        dx, dy, 
        dx_other_ghost, dy_other_ghost # Added new features
    ]], columns=["px", "py", "gx", "gy", "dx", "dy", "dx_other_ghost", "dy_other_ghost"])
    
    try:
        pred_numeric = model.predict(features)[0]
        return inv_move_map[pred_numeric] # Convert numeric prediction back to string
    except Exception as e:
        # print(f"Prediction failed: {e}. Falling back to default.")
        return "UP" # Fallback if prediction fails (e.g., model not trained)

def ghost_ai_move(ghost, player_pos, other_ghost_pos, model):
    """
    Determines the ghost's next move using the trained model for interception or A* fallback.
    """
    next_move_direction = None
    positions_to_avoid_for_ghost = {other_ghost_pos} if other_ghost_pos else set()
    
    if model:
        # Use the model to predict the player's next move based on *all* relevant ghost positions
        predicted_player_direction = predict_player_move(model, ghost.pos, player_pos, other_ghost_pos)
        
        # Calculate the predicted next position of the player
        predicted_player_next_pos = move(player_pos, predicted_player_direction)

        # Strategy 1: Try to intercept the predicted player's next position
        next_move_direction = a_star(ghost.pos, predicted_player_next_pos, positions_to_avoid_for_ghost)

        if next_move_direction:
            ghost.current_strategy = "Model-Intercept"
            # print(f"Ghost {ghost.id} using model to intercept. Predicted player move: {predicted_player_direction}")
        else:
            # If interception path is not found, fall back to chasing the current player position
            ghost.current_strategy = "A*-Chase"
            # print(f"Ghost {ghost.id} couldn't intercept, falling back to chase.")
            next_move_direction = a_star(ghost.pos, player_pos, positions_to_avoid_for_ghost)
    
    if not next_move_direction: # Fallback if no model or A* path found (or A* interception failed)
        if model: # If model was available but A* chase failed too
            ghost.current_strategy = "A*-Chase (Fallback)"
            # print(f"Ghost {ghost.id} no model or A* path, trying direct chase.")
            next_move_direction = a_star(ghost.pos, player_pos, positions_to_avoid_for_ghost)
        else: # No model at all, directly go for A* chase
            ghost.current_strategy = "A*-Chase (No Model)" # Clarify strategy when no model
            next_move_direction = a_star(ghost.pos, player.pos, positions_to_avoid_for_ghost)


    if not next_move_direction: # If even direct chase A* fails (unlikely in open maze)
        ghost.current_strategy = "Random"
        # print(f"Ghost {ghost.id} A* failed, trying random move.")
        # Last resort: random valid move
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
def ai_player_move(player, current_dots, ghost1, ghost2):
    if not current_dots:
        return None # No dots left

    min_score = float('inf')
    target_dot = None
    
    # Find the "safest" nearest dot
    for dot in current_dots:
        dist_to_dot = manhattan_distance(player.pos, dot)
        
        # Consider distance from dot to ghosts
        dist_dot_to_ghost1 = manhattan_distance(dot, ghost1.pos)
        dist_dot_to_ghost2 = manhattan_distance(dot, ghost2.pos)
        
        # Heuristic: prefer dots further from ghosts, closer to player
        # A lower "risk_score" is better.
        risk_score = dist_to_dot 
        
        # Penalize dots that are too close to ghosts
        # The penalty factor (20) can be tuned.
        if dist_dot_to_ghost1 < 5: # If ghost1 is close to the dot
            risk_score += (5 - dist_dot_to_ghost1) * 20 
        if dist_dot_to_ghost2 < 5: # If ghost2 is close to the dot
            risk_score += (5 - dist_dot_to_ghost2) * 20

        if risk_score < min_score:
            min_score = risk_score
            target_dot = dot
            
    if not target_dot: # Should not happen if dots exist, but as a safeguard
        target_dot = list(current_dots)[0] # Pick any if logic fails

    # Player avoids both ghosts when pathfinding
    ghost_positions_set = {ghost1.pos, ghost2.pos}
    direction_to_move = a_star(player.pos, target_dot, ghost_positions_set)
    
    # If no path to the target dot (e.g., surrounded by walls/ghosts), try a random valid move
    if not direction_to_move:
        possible_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
        np.random.shuffle(possible_moves)
        for move_dir in possible_moves:
            new_pos = move(player.pos, move_dir)
            # Ensure random move doesn't lead into a ghost
            if new_pos != player.pos and new_pos not in ghost_positions_set:
                return move_dir
        return None # If no valid random move, stay put
        
    return direction_to_move # This is a direction string like "UP"


# --- Main Game Reset Function ---
def reset_game():
    global score, dots, player, ghost1, ghost2, maze # Ensure 'maze' is global

    # 1. Reset the 'maze' to its original state from 'maze_layout'
    maze[:] = [list(row) for row in maze_layout] # Use slice assignment to modify the global list in place

    # 2. Reset score
    score = 0

    # 3. Reset and populate dots based on the original 'maze_layout'
    dots = set()
    for y_idx, row_str in enumerate(maze_layout):
        for x_idx, tile_char in enumerate(row_str):
            if tile_char == '.': # Only actual pellets defined by '.'
                dots.add((x_idx, y_idx))

    # 4. Find initial positions for player and ghosts using the freshly reset 'maze'
    player_start_pos = find_tile("P")
    ghost1_start_pos = find_tile("G")
    ghost2_start_pos = find_tile("G2") # Use 'G2' as defined in maze_layout

    # 5. Create player and ghost objects at their starting positions
    player = Player(player_start_pos)
    ghost1 = Ghost(ghost1_start_pos, RED, 1)
    ghost2 = Ghost(ghost2_start_pos, PINK, 2) # Uses PINK color

    # 6. Modify the current game's 'maze' (the working copy) for gameplay.
    # After finding their positions, change 'P', 'G', 'G2' in the 'maze' array to '.' (empty path).
    if player_start_pos != (1, 1): # Check if valid starting pos was found
        maze[player_start_pos[1]][player_start_pos[0]] = '.'
    if ghost1_start_pos != (1, 1):
        maze[ghost1_start_pos[1]][ghost1_start_pos[0]] = '.'
    if ghost2_start_pos != (1, 1):
        maze[ghost2_start_pos[1]][ghost2_start_pos[0]] = '.'


# --- Main Training Loop ---
username = get_username()
game_count = 0
iteration = 0 # Using 'iteration' for the overall training loop count
# Declare player, ghost1, ghost2, score, dots globally once if they are set in reset_game
player = None
ghost1 = None
ghost2 = None
score = 0
dots = set()

# Initialize models outside the loop to retain them after the loop ends
model1 = None
model2 = None

# Define the full set of feature names, including the new ones
feature_names = ["px", "py", "gx", "gy", "dx", "dy", "dx_other_ghost", "dy_other_ghost"]
class_names = [inv_move_map[i] for i in sorted(inv_move_map.keys())] # UP, DOWN, LEFT, RIGHT

while iteration < 500: # Running for 500 game iterations to collect lots of data
    game_count += 1
    iteration += 1
    print(f"\n--- Starting Game {game_count} (Overall Iteration {iteration}) ---")
    reset_game() # Resets maze, player, ghosts, dots, score
    countdown_timer()

    log1 = []
    log2 = []
    csv_file1 = data_filename_template.format(1, username)
    csv_file2 = data_filename_template.format(2, username)
    model_file1 = model_filename_template.format(1, username)
    model_file2 = model_filename_template.format(2, username)

    # --- Model Loading and Training Logic ---
    # Try to load existing models first
    if os.path.exists(model_file1):
        try:
            model1 = joblib.load(model_file1)
            print(f"Loaded existing Ghost 1 model from {model_file1}.")
        except Exception as e:
            print(f"Error loading Ghost 1 model from {model_file1}: {e}. Will attempt to train from CSV.")
            model1 = None # Reset model if loading fails
    
    # If model1 was not loaded or failed to load, train it from CSV data
    if model1 is None and os.path.exists(csv_file1):
        try:
            df = pd.read_csv(csv_file1)
            # print(f"Ghost 1 CSV entries: {len(df)}")
            if len(df) >= 1: # Need at least one sample to train
                # Ensure all required columns are present in the DataFrame
                if all(col in df.columns for col in feature_names + ["move"]):
                    X = df[feature_names]
                    y = df["move"]
                    if not X.empty and not y.empty:
                        # Changed to RandomForestClassifier
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
            model2 = None # Reset model if loading fails

    # If model2 was not loaded or failed to load, train it from CSV data
    if model2 is None and os.path.exists(csv_file2):
        try:
            df = pd.read_csv(csv_file2)
            # print(f"Ghost 2 CSV entries: {len(df)}")
            if len(df) >= 1:
                if all(col in df.columns for col in feature_names + ["move"]):
                    X = df[feature_names]
                    y = df["move"]
                    if not X.empty and not y.empty:
                        # Changed to RandomForestClassifier
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
    last_player_move_time = pygame.time.get_ticks() # Renamed for clarity
    last_ghost_move_time = pygame.time.get_ticks()

    while not game_over:
        screen.fill(BLACK)
        draw_maze(dots) # Pass the current 'dots' set

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                # Save data before exiting
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

        # Player (AI) move
        if current_time - last_player_move_time > player_move_delay:
            player_action_str = ai_player_move(player, dots, ghost1, ghost2)
            if player_action_str: # If a valid move string is returned
                # Log data before player moves, based on current positions
                # Relative positions for ghost 1
                dx1 = player.pos[0] - ghost1.pos[0]
                dy1 = player.pos[1] - ghost1.pos[1]
                # Relative positions of ghost 2 to player (for ghost 1's data)
                dx_other_ghost_for_g1 = player.pos[0] - ghost2.pos[0]
                dy_other_ghost_for_g1 = player.pos[1] - ghost2.pos[1]

                log1.append({
                    "px": player.pos[0], "py": player.pos[1],
                    "gx": ghost1.pos[0], "gy": ghost1.pos[1],
                    "dx": dx1, "dy": dy1,
                    "dx_other_ghost": dx_other_ghost_for_g1, # New feature
                    "dy_other_ghost": dy_other_ghost_for_g1, # New feature
                    "move": move_map[player_action_str] # Log the chosen action
                })

                # Relative positions for ghost 2
                dx2 = player.pos[0] - ghost2.pos[0]
                dy2 = player.pos[1] - ghost2.pos[1]
                # Relative positions of ghost 1 to player (for ghost 2's data)
                dx_other_ghost_for_g2 = player.pos[0] - ghost1.pos[0]
                dy_other_ghost_for_g2 = player.pos[1] - ghost1.pos[1]

                log2.append({
                    "px": player.pos[0], "py": player.pos[1],
                    "gx": ghost2.pos[0], "gy": ghost2.pos[1],
                    "dx": dx2, "dy": dy2,
                    "dx_other_ghost": dx_other_ghost_for_g2, # New feature
                    "dy_other_ghost": dy_other_ghost_for_g2, # New feature
                    "move": move_map[player_action_str] # Log the chosen action
                })
                
                player.move(player_action_str)
            last_player_move_time = current_time


        # Ghosts move
        if current_time - last_ghost_move_time > ghost_move_delay:
            # Pass the other ghost's position to avoid self-collision and for prediction feature
            ghost_ai_move(ghost1, player.pos, ghost2.pos, model1)
            ghost_ai_move(ghost2, player.pos, ghost1.pos, model2)
            last_ghost_move_time = current_time

        # Check for collisions with ghosts
        if player.pos == ghost1.pos or player.pos == ghost2.pos:
            game_over = True
            print("AI caught by a ghost!")


        # Player eats dot
        if player.pos in dots:
            dots.remove(player.pos)
            score += 10

        # Check for win condition (all dots eaten)
        if not dots:
            screen.fill(BLACK)
            draw_text("AI Wins!", (WIDTH // 2, HEIGHT // 2 - 50), font_input, YELLOW)
            draw_text(f"Score: {score}", (WIDTH // 2, HEIGHT // 2 + 50), font_input, YELLOW)
            pygame.display.flip()
            pygame.time.delay(2000) # Display win message longer
            game_over = True
            print("AI wins! All dots collected.")

        # Draw everything
        player.draw()
        ghost1.draw()
        ghost2.draw()
        
        # Display ghost strategies
        draw_text(f"G1: {ghost1.current_strategy}", (WIDTH - 120, TILE_SIZE * 2 - 10), font_ghost_info, RED)
        draw_text(f"G2: {ghost2.current_strategy}", (WIDTH - 120, TILE_SIZE * 2 + 10), font_ghost_info, PINK)

        draw_text(f"Player: AI ({username})  Score: {score}  Game: {game_count} Iter: {iteration}",
                    (WIDTH // 2, TILE_SIZE // 2), font_game, WHITE) # Adjusted Y for top display

        pygame.display.flip()
        clock.tick(60) # Target 60 FPS

    # --- End of single game loop ---

    # Save movement data collected during the game
    if log1:
        df1_new = pd.DataFrame(log1)
        if os.path.exists(csv_file1):
            try:
                old_df1 = pd.read_csv(csv_file1)
                df1_combined = pd.concat([old_df1, df1_new], ignore_index=True)
            except pd.errors.EmptyDataError: # Handle case where existing CSV is empty
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

    # Show game over screen briefly if not a win
    if not dots and game_over: # AI Wins case already handled
        pass
    elif game_over: # AI was caught
        screen.fill(BLACK)
        draw_text("Game Over!", (WIDTH // 2, HEIGHT // 2 - 100), font_input, RED)
        if player.pos == ghost1.pos or player.pos == ghost2.pos :
            draw_text("AI was caught!", (WIDTH // 2, HEIGHT // 2 - 50), font_input, RED)
        draw_text(f"Final Score: {score}", (WIDTH // 2, HEIGHT // 2 + 50), font_input, WHITE)
        pygame.display.flip()
        pygame.time.delay(2000) # Display game over message

# --- Save trained models after the training loop completes ---
# It's better to save the model *after* the entire training loop
# so it has learned from all iterations.
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