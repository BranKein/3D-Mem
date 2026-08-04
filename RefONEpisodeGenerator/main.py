"""RefONEpisodeGenerator entry point.

Usage examples:
  # command 1: generate instruction style lists
  python main.py generate -o out/run1 --num-scenes 5 --episodes-per-scene 10

  # command 2: convert generated style lists into real HM3D episode json (needs habitat-sim)
  python main.py build -i out/run1 \
      --scenes "hm3d/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb,..."

  # auxiliary: validate
  python main.py validate -i out/run1
"""
import sys

from refon.cli import main

if __name__ == "__main__":
    sys.exit(main())
