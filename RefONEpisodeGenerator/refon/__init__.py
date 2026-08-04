"""RefONEpisodeGenerator

A pipeline that generates episodes for an HM3D-based Referential
Object-goal Navigation (MultiON) benchmark.

It splits into two stages:
  1. instruction style list generation (structure only, scene-independent) -> refon.balancer
  2. style list -> real HM3D episode json (scene/object assignment)        -> refon.episode_builder

The core design is OOP: each instruction style (role) is a class that owns its
admissibility (constraints), automaton transition, object resolution, and
instruction rendering. A new style is added by subclassing InstructionStyle and
registering it; the pipeline picks it up automatically.
"""

from .automaton import AR_EXPECT_ONE, AR_EXPECT_POST, AR_NONE, GenContext
from .style_token import StyleList, StyleToken
from .config import GeneratorConfig

__all__ = [
    "GenContext",
    "AR_NONE",
    "AR_EXPECT_ONE",
    "AR_EXPECT_POST",
    "StyleToken",
    "StyleList",
    "GeneratorConfig",
]
