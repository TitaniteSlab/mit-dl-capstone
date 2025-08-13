# Utility script to generate a list of legal moves for the output layer of the neural network.
import json

FILES = "abcdefgh"
RANKS = "12345678"

# Helpers
def in_board(f, r):
    return 0 <= f < 8 and 0 <= r < 8

def sq_name(f, r):
    return FILES[f] + RANKS[r]

# AlphaZero-like policy planes:
# 56 sliding: 8 directions × steps 1..7
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

# 8 knight jumps
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

# 9 underpromotion planes: promotions to N,B,R × directions
# Directions are from side-to-move perspective (White forward).
# For Black to move, boards are typically flipped in net input so this remains consistent.
UNDERPROMO_PIECES = ["n", "b", "r", "q"]
UNDERPROMO_DIRS = [
    (0, 1, "forward"),
    (1, 1, "capture-right"),
    (-1, 1, "capture-left")
]

# Build the 4,672 indexed list
moves = []
index = 0

# For each from-square (side-to-move perspective)
for r_from in range(8):
    for f_from in range(8):
        from_sq = sq_name(f_from, r_from)

        # 56 sliding planes
        for dx, dy in SLIDING_DIRS:
            for step in range(1, 8):
                f_to = f_from + dx * step
                r_to = r_from + dy * step
                if not in_board(f_to, r_to):
                    break
                to_sq = sq_name(f_to, r_to)
                moves.append({
                    "index": index,
                    "kind": "slide",
                    "from": from_sq,
                    "to": to_sq,
                    "dir": [dx, dy],
                    "steps": step,
                    "promotion": None
                })
                index += 1

        # 8 knight planes
        for dx, dy in KNIGHT_DELTAS:
            f_to = f_from + dx
            r_to = r_from + dy
            if in_board(f_to, r_to):
                to_sq = sq_name(f_to, r_to)
                moves.append({
                    "index": index,
                    "kind": "knight",
                    "from": from_sq,
                    "to": to_sq,
                    "delta": [dx, dy],
                    "promotion": None
                })
                index += 1

        # 12 promotion planes (N, B, R, Q) × (forward, capture-right, capture-left)
        for promo in UNDERPROMO_PIECES:
            for dx, dy, dir_name in UNDERPROMO_DIRS:
                f_to = f_from + dx
                r_to = r_from + dy
                if r_from == 6 and r_to == 7 and in_board(f_to, r_to):
                    to_sq = sq_name(f_to, r_to)
                    moves.append({
                        "index": index,
                        "kind": "underpromotion",
                        "from": from_sq,
                        "to": to_sq,
                        "dir": dir_name,
                        "promotion": promo
                    })
                    index += 1

print(f"len(moves): {len(moves)}")

with open("data/moves_legal.json", "w") as f:
    json.dump(moves, f, indent=2)

print("Wrote data/moves_legal.json with", len(moves), "entries")

uci_moves = [m["from"] + m["to"] + (m["promotion"] if m["promotion"] else "") for m in moves]

with open("data/moves_legal_uci.txt", "w") as f:
    f.write("\n".join(uci_moves) + "\n")

print("Wrote data/moves_legal_uci.txt with", len(uci_moves), "entries")
