# Utility script to process FICS PGN files downloaded from https://www.ficsgames.org/
# Removes disconnection games and games shorter than 5 moves. Combines all months into a single file. Strips metadata.

def reduce_pgn(input_path: str, output_path: str) -> None:
  with open(input_path, 'r', encoding='utf-8') as inp, open(output_path, 'a', encoding='utf-8') as out:
    for line in inp:
      if line.startswith('['):
        continue
      if 'disconnection' in line:
        continue
      if '5.' not in line:
        continue
      stripped = line.strip()
      if stripped:
        out.write(stripped + '\n')

def main() -> None:
  output_path = 'data/fics-2024.pgn'
  for month in range(1, 13):
    input_path = f'data/fics-2000_2024-{month:02d}.pgn'
    reduce_pgn(input_path, output_path)

if __name__ == '__main__':
  main()
