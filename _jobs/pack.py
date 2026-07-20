#! /usr/bin/env python
import os
import subprocess
import sys

cells = [cell for cell in os.environ["CELLS"].split(";") if cell.strip()]
serial = os.environ.get("SERIAL") == "1"

failed, running = [], []
for cell in cells:
  command = ["python3", *cell.split()]
  print(f"start: {' '.join(command)}", flush=True)
  process = subprocess.Popen(command)
  if serial:
    failed += [cell] if process.wait() != 0 else []
  else:
    running.append((cell, process))

failed += [cell for cell, process in running if process.wait() != 0]
for cell in failed:
  print(f"failed: {cell}", file=sys.stderr)
sys.exit(1 if failed else 0)
