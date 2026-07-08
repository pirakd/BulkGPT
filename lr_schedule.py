"""Iteration-budget-independent learning rate scheduling.

The previous approach annealed LR with a single cosine decay over
`lr_decay_iters`, which was fixed to whatever `max_iters` happened to be the
first time a run started, then carried forward through every `--resume`.
Extending `max_iters` later (or resuming many times) didn't reshape that
schedule -- but it also meant that once the original horizon was reached, LR
just sat flat at `min_lr` forever, regardless of how much more training
happened afterward. Whether the schedule "still makes sense" ended up
depending entirely on a number picked once, early on, with no way to see
that from the config.

An intermediate version replaced that with cosine annealing with warm
restarts (SGDR): the LR would periodically jump back up to (a decayed
fraction of) its peak and re-anneal. In practice each restart's LR spike
forced the model to re-explore and re-converge, which is expensive in wall
clock/compute for very little benefit over just holding a low LR steady.

This module instead uses a much cheaper, monotonic schedule: linear warmup,
then a flat LR that only ever decreases, and only when validation loss
plateaus (`ReduceLROnPlateauLR`, mirroring `torch.optim.lr_scheduler.
ReduceLROnPlateau`). There are no restarts and no LR spikes -- once training
has converged enough that val loss stops improving, the LR is halved (or by
whatever `factor` you configure) and training continues from there, with a
cooldown before another reduction can trigger. It still never depends on
`max_iters`, and it still floors at `min_lr` indefinitely.

All schedule state (current LR, best val, patience/cooldown counters) is
plain-data and serializable via `state_dict()` / `load_state_dict()`, so it
should be saved in and restored from the training checkpoint alongside the
model/optimizer state -- resuming a run continues the schedule exactly where
it left off.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class LRScheduleConfig:
    """Config for `ReduceLROnPlateauLR`. All fields are iteration counts /
    LR values, never tied to `max_iters`."""

    base_lr: float
    min_lr: float
    warmup_iters: int = 100
    # Multiply LR by this factor whenever a plateau is detected. Must be in
    # (0, 1]; 1.0 effectively disables reductions.
    factor: float = 0.5
    # Number of consecutive `on_eval` calls with no meaningful val-loss
    # improvement before the LR is reduced.
    patience: int = 5
    # Minimum decrease in val loss required to reset the plateau counter.
    min_delta: float = 0.0
    # Number of `on_eval` calls to wait after a reduction before another
    # reduction can trigger, even if the plateau counter would allow it.
    cooldown: int = 0


class ReduceLROnPlateauLR:
    """Linear warmup followed by a flat LR that only decreases, triggered by
    a validation-loss plateau. No restarts, no LR spikes.

    Typical usage::

        scheduler = ReduceLROnPlateauLR(cfg.lr_schedule)
        if resuming:
            scheduler.load_state_dict(ckpt["lr_schedule_state"])

        for it in range(...):
            lr = scheduler.step(it)          # advances + returns current LR
            ... train step using `lr` ...
            if it % eval_interval == 0:
                val_loss = evaluate(...)
                scheduler.on_eval(val_loss)  # may reduce LR

        ckpt["lr_schedule_state"] = scheduler.state_dict()
    """

    def __init__(self, cfg: LRScheduleConfig):
        self.cfg = cfg
        self._lr = cfg.base_lr
        self._best_val = float("inf")
        self._plateau_count = 0
        self._cooldown_count = 0

    def lr_at(self, it: int) -> float:
        """Look up the LR for iteration `it` under the current schedule
        state, without mutating anything. Use `step()` in the training loop
        instead so warmup is applied correctly."""
        if self.cfg.warmup_iters > 0 and it < self.cfg.warmup_iters:
            return self._lr * (it + 1) / self.cfg.warmup_iters
        return self._lr

    def step(self, it: int) -> float:
        """Return the LR to use for iteration `it`. Call this once per
        training iteration."""
        return self.lr_at(it)

    def on_eval(self, val_loss: float) -> bool:
        """Report a validation loss. Returns True if this call reduces the
        LR (takes effect on the very next `step()` call)."""
        if val_loss < self._best_val - self.cfg.min_delta:
            self._best_val = val_loss
            self._plateau_count = 0
            return False

        if self._cooldown_count > 0:
            self._cooldown_count -= 1
            return False

        self._plateau_count += 1
        if self._plateau_count >= self.cfg.patience:
            self._lr = max(self.cfg.min_lr, self._lr * self.cfg.factor)
            self._plateau_count = 0
            self._cooldown_count = self.cfg.cooldown
            return True
        return False

    def state_dict(self) -> dict:
        return {
            "lr": self._lr,
            "best_val": self._best_val,
            "plateau_count": self._plateau_count,
            "cooldown_count": self._cooldown_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self._lr = state["lr"]
        self._best_val = state["best_val"]
        self._plateau_count = state["plateau_count"]
        self._cooldown_count = state.get("cooldown_count", 0)
