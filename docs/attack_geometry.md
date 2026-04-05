# Attack Geometry Plan

This note translates the "AlphaFold in front of MCTS" idea into the current
TeenyZero codebase.

## What This Should Mean Here

In this repo, the right analogue is not "replace MCTS with a biology model."
It is:

1. build a richer board representation before policy/value prediction
2. train that representation to model piece relations, attack structure, and
   king pressure
3. let MCTS consume sharper priors and values so fewer simulations are needed

So the new component belongs between board encoding and the policy/value heads,
not as a separate engine bolted onto tree search.

## Current Seam In The Code

Today the stack is:

- `teenyzero/mcts/evaluator.py`
  - encodes a board into `(planes, 8, 8)`
  - runs the model
  - masks logits down to legal moves
- `teenyzero/alphazero/model.py`
  - plain residual CNN trunk
  - policy head -> 4672 move logits
  - value head -> scalar
- `teenyzero/mcts/search.py`
  - uses priors and value only

That is a good seam. We can keep MCTS mostly unchanged if the evaluator still
returns `MovePriors` plus a scalar value.

## Recommended V1

Start with a square-relation model, not a full AlphaFold clone.

Use:

1. square tokens
   - 64 tokens, one per board square
   - each token gets local features from the existing planes
2. pair bias / relation features
   - same file
   - same rank
   - knight relation
   - sliding ray relation
   - occupied by friendly / enemy / empty
   - attacked by friendly / enemy
   - in friendly king zone / enemy king zone
3. a few relation blocks
   - token update from square context
   - pair-conditioned attention or pair bias
   - no need for expensive triangle updates in v1
4. policy/value heads on top
   - policy head can still emit 4672 logits
   - value stays scalar

This gives us something AlphaFold-inspired in spirit:

- single representation for "entities"
- explicit pairwise structure
- learned geometry before the final decision heads

But it stays cheap enough for self-play.

## Why Not Full Move-Pair Modeling First

A move-to-move relation model sounds appealing, but it is the wrong first
object here.

- The legal move set changes every position.
- The current policy target is indexed in a fixed 4672-space.
- MCTS expansion is organized around board states, not a stable move graph.

The stable objects are squares and pieces. Train geometry there first, then map
that geometry into move logits.

## Training Signal Problem

Right now training only supervises:

- search policy target `pi`
- game outcome target `z`

That is enough for AlphaZero, but weak for an explicit geometry module.

If we want the model to learn board structure on purpose, add 1-2 auxiliary
targets derived from the board:

1. attack map target
   - predict squares attacked by the side to move
   - optionally also attacked by the opponent
2. king zone pressure target
   - predict pressure on the 8-neighborhood around each king
3. optional pinned / hanging target
   - only if the first two are stable and useful

Those targets are cheap to compute from `python-chess` during self-play and do
not require extra labeling infrastructure.

## First Implementation Phases

### Phase 1: Clean Architecture Hook

Refactor model construction so architecture choice is explicit by
`MODEL_VERSION`, instead of always building `AlphaNet`.

Goal:

- keep current CNN as one implementation
- add a second experimental relational model without breaking old checkpoints

### Phase 2: Auxiliary Geometry Labels

Extend replay collection so each saved position can optionally include:

- friendly attack map
- enemy attack map
- king-zone pressure

Keep this versioned behind a new replay encoder version.

### Phase 3: Relational Trunk

Add a new model variant:

- plane encoder -> 64 square embeddings
- learned relation bias between squares
- 2-4 lightweight relation blocks
- policy/value heads
- optional auxiliary heads for geometry targets

### Phase 4: MCTS Integration

Keep `teenyzero/mcts/search.py` mostly unchanged at first.

Only add new search behavior if the model proves strong enough, for example:

- root move pruning by geometry confidence
- dynamic simulation budgets
- geometry-guided FPU or prior sharpening

Do not do this before the model itself is measurably better.

## What Success Should Look Like

The objective is not "cool latent structure."
It is:

1. stronger policy quality at low simulation counts
2. stronger value estimates in tactical positions
3. same or better playing strength per unit wall-clock time

The key comparison should be:

- old model at `N` simulations
- relational model at `N / 2` or `N / 4` simulations

If the new model needs the same wall-clock budget to get the same strength,
then it did not solve the actual search problem.

## Recommended First Concrete Build

If continuing from here, the best next coding step is:

1. split model building by `MODEL_VERSION`
2. add a new experimental relational model class alongside `AlphaNet`
3. keep the evaluator and MCTS output contract unchanged

That gives us a safe place to iterate on architecture without destabilizing the
rest of the training loop.
