# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file contains logic for a competitive Battlesnake implementation.
# For more info see docs.battlesnake.com

import random
import typing
from typing import List, Dict, Tuple, Set
from collections import deque

# Move vectors for cardinal directions
MOVES = {
    "up": {"x": 0, "y": 1},
    "down": {"x": 0, "y": -1},
    "left": {"x": -1, "y": 0},
    "right": {"x": 1, "y": 0}
}


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "",  # TODO: Your Battlesnake Username
        "color": "#FF3366",
        "head": "broad",
        "tail": "freckled",
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


def get_neighbors(pos: Dict[str, int]) -> List[Tuple[int, int]]:
    """Get all neighboring coordinates."""
    return [
        (pos["x"] + 1, pos["y"]),
        (pos["x"] - 1, pos["y"]),
        (pos["x"], pos["y"] + 1),
        (pos["x"], pos["y"] - 1)
    ]


def is_safe(pos: Tuple[int, int], width: int, height: int,
             obstacles: Set[Tuple[int, int]]) -> bool:
    """Check if a position is safe (in bounds and not blocked)."""
    x, y = pos
    return 0 <= x < width and 0 <= y < height and pos not in obstacles


def flood_fill_safe(pos: Tuple[int, int], width: int, height: int,
                    obstacles: Set[Tuple[int, int]],
                    min_space: int = 10) -> bool:
    """Check if we have enough space using flood fill algorithm."""
    visited = set()
    stack = [pos]
    max_iterations = min_space * 10  # Prevent infinite loops

    while stack and len(visited) < min_space and len(visited) < max_iterations:
        curr = stack.pop()
        if curr in visited or curr in obstacles:
            continue
        visited.add(curr)
        x, y = curr
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                stack.append((nx, ny))

    return len(visited) >= min_space


def evaluate_move(pos: Tuple[int, int], food_pos: Tuple[int, int],
                  board_center: Tuple[int, int]) -> float:
    """Score a position based on food proximity and center control."""
    dist_to_food = abs(pos[0] - food_pos[0]) + abs(pos[1] - food_pos[1])
    dist_to_center = abs(pos[0] - board_center[0]) + abs(pos[1] - board_center[1])
    # Lower distances are better, so we negate
    return -dist_to_food - dist_to_center * 0.3


def move(game_state: typing.Dict) -> typing.Dict:
    """Choose the best move for the current turn."""
    my_snake = game_state["you"]
    my_head = my_snake["head"]
    board = game_state["board"]

    width = board["width"]
    height = board["height"]
    my_health = my_snake["health"]
    my_length = my_snake["length"]

    # Collect all obstacles (bodies excluding moving tails)
    all_body_parts = set()
    for snake in board["snakes"]:
        for segment in snake["body"][:-1]:
            all_body_parts.add((segment["x"], segment["y"]))

    # Add hazards
    hazards = set((h["x"], h["y"]) for h in board.get("hazards", []))
    combined_obstacles = all_body_parts | hazards

    # Food positions
    food = [(f["x"], f["y"]) for f in board["food"]]

    # Current positions
    my_head_pos = (my_head["x"], my_head["y"])
    my_neck_pos = (my_snake["body"][1]["x"], my_snake["body"][1]["y"])

    # Build opponent head positions for collision detection
    opponent_heads = set()
    for snake in board["snakes"]:
        if snake["id"] != my_snake["id"]:
            opponent_heads.add((snake["head"]["x"], snake["head"]["y"]))

    safe_moves = []

    # Evaluate all possible moves
    for direction, move_vec in MOVES.items():
        new_x = my_head_pos[0] + move_vec["x"]
        new_y = my_head_pos[1] + move_vec["y"]
        new_pos = (new_x, new_y)

        # Don't move backwards (into neck)
        if (new_x, new_y) == my_neck_pos:
            continue

        # Check bounds, hazards, body collisions
        if not is_safe(new_pos, width, height, combined_obstacles):
            continue

        # Check head-to-head collision risk
        # Avoid moving into adjacent cells of equal or longer snakes
        collision_risk = False
        for snake in board["snakes"]:
            if snake["id"] != my_snake["id"]:
                opp_head = (snake["head"]["x"], snake["head"]["y"])
                opp_neighbors = get_neighbors(snake["head"])
                if new_pos in opp_neighbors and snake["length"] >= my_length:
                    collision_risk = True
                    break

        if collision_risk:
            continue

        # Check space availability using flood fill
        # Use larger space requirement for end-game survival
        space_needed = 15 if len(board["snakes"]) <= 3 else 10
        if not flood_fill_safe(new_pos, width, height, combined_obstacles, min_space=space_needed):
            continue

        safe_moves.append((direction, new_pos))

    # If no safe moves, pick any non-backtracking direction
    if not safe_moves:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    # Strategy selection based on game state
    best_move = None
    best_score = float('-inf')
    board_center = (width // 2, height // 2)

    # Health threshold for aggressive food seeking
    need_food = my_health < 50

    if need_food and food:
        # Find nearest food and move towards it
        for direction, pos in safe_moves:
            nearest_food = min(food, key=lambda f: abs(pos[0] - f[0]) + abs(pos[1] - f[1]))
            score = evaluate_move(pos, nearest_food, board_center)
            if score > best_score:
                best_score = score
                best_move = direction
    else:
        # When healthy: prefer moves with more space, stay in center
        for direction, pos in safe_moves:
            # Score based on flood fill space available
            space_available = flood_fill_safe(pos, width, height, combined_obstacles, min_space=50)
            center_dist = abs(pos[0] - board_center[0]) + abs(pos[1] - board_center[1])
            score = space_available * 10 - center_dist
            if score > best_score:
                best_score = score
                best_move = direction

    print(f"MOVE {game_state['turn']}: {best_move} (health: {my_health}, snakes: {len(board['snakes'])})")
    return {"move": best_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
