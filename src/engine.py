#!/usr/bin/env python3
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import chess
import chess.pgn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_FILE = './model.pt'
MOVES_FILE = '../data/moves.txt'

model = None
move_to_index = None
index_to_move = None

warnings.filterwarnings('ignore')

class ChessCNN(nn.Module):
  def __init__(self, num_moves: int):
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv2d(18, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU()
    )
    self.flatten = nn.Flatten()
    self.classifier = nn.Sequential(
      nn.Linear(128 * 2 * 2, 512),
      nn.ReLU(),
      nn.Linear(512, num_moves)
    )
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = self.flatten(x)
    x = self.classifier(x)
    return x


# First we parse PGN into a python-chess board, then this encodes it into a 8x8x18 tensor for input to the model
def encode_board(board: 'chess.Board') -> torch.Tensor:
  planes = np.zeros((18, 8, 8), dtype=np.float32)
  piece_type_to_index = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
  }
  for square, piece in board.piece_map().items():
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    base = piece_type_to_index[piece.piece_type]
    channel = base if piece.color == chess.WHITE else base + 6
    planes[channel, 7 - rank, file] = 1.0
  if board.turn == chess.WHITE:
    planes[12, :, :] = 1.0
  if board.has_kingside_castling_rights(chess.WHITE):
    planes[13, :, :] = 1.0
  if board.has_queenside_castling_rights(chess.WHITE):
    planes[14, :, :] = 1.0
  if board.has_kingside_castling_rights(chess.BLACK):
    planes[15, :, :] = 1.0
  if board.has_queenside_castling_rights(chess.BLACK):
    planes[16, :, :] = 1.0
  if board.ep_square is not None:
    r = chess.square_rank(board.ep_square)
    f = chess.square_file(board.ep_square)
    planes[17, 7 - r, f] = 1.0
  return torch.from_numpy(planes)

def load_moves():
  global move_to_index, index_to_move
  with open(MOVES_FILE, 'r', encoding='utf-8') as f:
    ALL_MOVES = [line.strip() for line in f.readlines() if line.strip()]
  move_to_index = {m: i for i, m in enumerate(ALL_MOVES)}
  index_to_move = ALL_MOVES

def load_model():
  global model
  if index_to_move is None:
    load_moves()
  num_moves = len(index_to_move)
  m = ChessCNN(num_moves).to(device)
  state = torch.load(MODEL_FILE, map_location=device, weights_only=True)
  if isinstance(state, dict) and 'model_state_dict' in state:
    sd = state['model_state_dict']
  elif isinstance(state, dict) and 'state_dict' in state:
    sd = state['state_dict']
  elif isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
    sd = state['model']
  else:
    sd = state
  try:
    m.load_state_dict(sd)
  except Exception:
    m.load_state_dict(sd, strict=False)
  model = m.eval()

def predict_best_move(board: 'chess.Board') -> str:
  x = encode_board(board).unsqueeze(0).to(device)
  with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    sorted_indices = torch.argsort(probs, dim=0, descending=True).cpu().tolist()
    for idx in sorted_indices:
      uci = index_to_move[idx]
      try:
        move = chess.Move.from_uci(uci)
      except Exception:
        continue
      if board.is_legal(move):
        return uci
  return '0000'

class UCIEngine:
  def __init__(self):
    self.board = chess.Board()
    self.stop_requested = False

  def reset(self):
    self.board = chess.Board()
    self.stop_requested = False

  def cmd_uci(self):
    print('id name NNChess')
    print('id author Anthony')
    print('uciok')
    sys.stdout.flush()

  def cmd_isready(self):
    print('readyok')
    sys.stdout.flush()

  def cmd_ucinewgame(self):
    self.reset()

  def cmd_setoption(self, _line: str):
    pass

  def cmd_position(self, line: str):
    tokens = line.strip().split()
    if len(tokens) < 2:
      return
    if 'startpos' in tokens:
      self.board = chess.Board()
      if 'moves' in tokens:
        mi = tokens.index('moves')
        for u in tokens[mi + 1:]:
          try:
            self.board.push_uci(u)
          except Exception:
            break
      return
    if 'fen' in tokens:
      fi = tokens.index('fen')
      if 'moves' in tokens:
        mi = tokens.index('moves')
        fen = ' '.join(tokens[fi + 1:mi])
        try:
          self.board.set_fen(fen)
        except Exception:
          return
        for u in tokens[mi + 1:]:
          try:
            self.board.push_uci(u)
          except Exception:
            break
      else:
        fen = ' '.join(tokens[fi + 1:fi + 7])
        try:
          self.board.set_fen(fen)
        except Exception:
          return

  def cmd_go(self, line: str):
    if self.board.is_game_over():
      print('bestmove 0000')
      sys.stdout.flush()
      return
    uci = predict_best_move(self.board)
    print(f'bestmove {uci}')
    sys.stdout.flush()

  def cmd_stop(self):
    self.stop_requested = True

def main():
  print('# Engine startup...')
  torch.set_grad_enabled(False)
  print('# Loading moves...')
  load_moves()
  print('# Loading model...')
  load_model()
  print('# Waiting for commands...')
  engine = UCIEngine()
  for raw in sys.stdin:
    line = raw.strip()
    if not line:
      continue
    #print(f'# Received command: {line}')
    if line == 'uci':
      engine.cmd_uci()
    elif line == 'isready':
      engine.cmd_isready()
    elif line == 'ucinewgame':
      engine.cmd_ucinewgame()
    elif line.startswith('setoption'):
      engine.cmd_setoption(line)
    elif line.startswith('position'):
      engine.cmd_position(line)
    elif line.startswith('go'):
      engine.cmd_go(line)
    elif line == 'stop':
      engine.cmd_stop()
    elif line == 'quit':
      break

if __name__ == '__main__':
  main()
