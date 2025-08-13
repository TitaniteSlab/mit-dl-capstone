import json

FILES = "abcdefgh"
RANKS = "12345678"

def in_board(f, r):
    return 0 <= f < 8 and 0 <= r < 8

def sq_name(f, r):
    return FILES[f] + RANKS[r]

SLIDING_DIRS = [
    (0, 1),   # N
    (0, -1),  # S
    (1, 0),   # E
    (-1, 0),  # W
    (1, 1),   # NE
    (-1, 1),  # NW
    (1, -1),  # SE
    (-1, -1)  # SW
]

KNIGHT_DELTAS = [
    (1, 2),
    (2, 1),
    (-1, 2),
    (-2, 1),
    (1, -2),
    (2, -1),
    (-1, -2),
    (-2, -1)
]

UNDERPROMO_PIECES = ["n", "b", "r"]
UNDERPROMO_DIRS = [
    (0, 1, "forward"),
    (1, 1, "capture-right"),
    (-1, 1, "capture-left")
]

moves = []
index = 0

for r_from in range(8):
    for f_from in range(8):
        from_sq = sq_name(f_from, r_from)

        # 56 sliding planes: include all 7 steps, even if off-board
        for dx, dy in SLIDING_DIRS:
            for step in range(1, 8):
                f_to = f_from + dx * step
                r_to = r_from + dy * step
                valid = in_board(f_to, r_to)
                to_sq = sq_name(f_to, r_to) if valid else None
                moves.append({
                    "index": index,
                    "kind": "slide",
                    "from": from_sq,
                    "to": to_sq,
                    "dir": [dx, dy],
                    "steps": step,
                    "promotion": None,
                    "valid": valid
                })
                index += 1

        # 8 knight planes
        for dx, dy in KNIGHT_DELTAS:
            f_to = f_from + dx
            r_to = r_from + dy
            valid = in_board(f_to, r_to)
            to_sq = sq_name(f_to, r_to) if valid else None
            moves.append({
                "index": index,
                "kind": "knight",
                "from": from_sq,
                "to": to_sq,
                "delta": [dx, dy],
                "promotion": None,
                "valid": valid
            })
            index += 1

        # 9 underpromotion planes: N/B/R × forward, capture-right, capture-left
        for promo in UNDERPROMO_PIECES:
            for dx, dy, dir_name in UNDERPROMO_DIRS:
                f_to = f_from + dx
                r_to = r_from + dy
                valid = in_board(f_to, r_to)
                to_sq = sq_name(f_to, r_to) if valid else None
                moves.append({
                    "index": index,
                    "kind": "underpromotion",
                    "from": from_sq,
                    "to": to_sq,
                    "dir": dir_name,
                    "promotion": promo,
                    "valid": valid
                })
                index += 1

assert len(moves) == 8 * 8 * 73, f"Expected 4672 moves, got {len(moves)}"

with open("alphazero_moves.json", "w") as f:
    json.dump(moves, f, indent=2)

print("Wrote alphazero_moves.json with", len(moves), "entries")
