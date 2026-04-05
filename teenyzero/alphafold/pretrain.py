from __future__ import annotations

import contextlib
import os
import time

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from teenyzero.alphafold.features import build_square_target_tensor, encode_board_state
from teenyzero.alphazero.checkpoints import load_checkpoint, save_checkpoint
from teenyzero.alphazero.runtime import get_runtime_profile


PROFILE = get_runtime_profile()


class RandomBoardPositionDataset(Dataset):
    def __init__(self, sample_count: int, min_plies: int = 6, max_plies: int = 80, rng_seed: int | None = None):
        self.sample_count = max(1, int(sample_count))
        self.min_plies = max(0, int(min_plies))
        self.max_plies = max(self.min_plies, int(max_plies))
        self.rng_seed = int(time.time() * 1000) if rng_seed is None else int(rng_seed)

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.rng_seed + int(idx))
        board = self._sample_board(rng)
        state = encode_board_state(board)
        targets = build_square_target_tensor(board)
        return torch.from_numpy(state).float(), torch.from_numpy(targets).float()

    def _sample_board(self, rng: np.random.Generator) -> chess.Board:
        for _ in range(12):
            board = chess.Board()
            target_plies = int(rng.integers(self.min_plies, self.max_plies + 1))
            for _ in range(target_plies):
                if board.is_game_over(claim_draw=True):
                    break
                move = self._sample_move(board, rng)
                board.push(move)
            if not board.is_game_over(claim_draw=True):
                return board
        return chess.Board()

    def _sample_move(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 1:
            return legal_moves[0]

        promotions = []
        captures = []
        checks = []
        quiets = []
        for move in legal_moves:
            if move.promotion:
                promotions.append(move)
                continue
            if board.is_capture(move):
                captures.append(move)
            elif board.gives_check(move):
                checks.append(move)
            else:
                quiets.append(move)

        roll = float(rng.random())
        if promotions and roll < 0.08:
            return promotions[int(rng.integers(0, len(promotions)))]
        if captures and roll < 0.38:
            return captures[int(rng.integers(0, len(captures)))]
        if checks and roll < 0.56:
            return checks[int(rng.integers(0, len(checks)))]
        pool = quiets or legal_moves
        return pool[int(rng.integers(0, len(pool)))]


class AlphaFoldPretrainer:
    def __init__(
        self,
        model,
        device: str = "cpu",
        lr: float | None = None,
        weight_decay: float | None = None,
        precision: str | None = None,
        use_compile: bool | None = None,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 0.0,
    ):
        self.device = device
        self.model = model.to(device)
        self.lr = float(PROFILE.train_lr if lr is None else lr)
        self.weight_decay = float(PROFILE.train_weight_decay if weight_decay is None else weight_decay)
        self.precision = (PROFILE.train_precision if precision is None else precision).lower()
        self.use_compile = bool(PROFILE.train_compile if use_compile is None else use_compile)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.max_grad_norm = float(max_grad_norm)
        self.use_channels_last = device == "cuda"
        self.use_autocast = self.device == "cuda" and self.precision in {"fp16", "bf16"}
        self.autocast_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        scaler_enabled = self.device == "cuda" and self.precision == "fp16"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        if self.device == "cuda":
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        if self.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if self.use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

    def _autocast_context(self):
        if not self.use_autocast:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def load_checkpoint(self, path: str):
        return load_checkpoint(self.model, path, map_location=self.device, allow_partial=True)

    def save_checkpoint(self, path: str, extra_meta=None):
        meta = {
            "model_version": 6,
            "architecture": "alphafold_board",
            "pretraining_stage": "geometry",
            "pretraining_objective": "attack_maps_and_king_pressure",
        }
        if extra_meta:
            meta.update(extra_meta)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_checkpoint(self.model, path, extra_meta=meta)

    def train_epoch(self, dataloader, progress_callback=None):
        self.model.train()
        total_loss = 0.0
        attack_loss_total = 0.0
        pressure_loss_total = 0.0
        batch_count = 0
        samples_seen = 0
        total_batches = len(dataloader)
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (states, targets) in enumerate(dataloader, start=1):
            states = states.to(self.device, non_blocking=self.device == "cuda")
            targets = targets.to(self.device, non_blocking=self.device == "cuda")
            if self.use_channels_last:
                states = states.contiguous(memory_format=torch.channels_last)
            samples_seen += int(states.size(0))

            with self._autocast_context():
                _, _, aux = self.model(states, return_aux=True)
                friendly_attack_loss = F.binary_cross_entropy_with_logits(aux["friendly_attack"], targets[:, 0])
                enemy_attack_loss = F.binary_cross_entropy_with_logits(aux["enemy_attack"], targets[:, 1])
                friendly_pressure_loss = F.mse_loss(torch.sigmoid(aux["friendly_king_pressure"]), targets[:, 2])
                enemy_pressure_loss = F.mse_loss(torch.sigmoid(aux["enemy_king_pressure"]), targets[:, 3])

                attack_loss = 0.5 * (friendly_attack_loss + enemy_attack_loss)
                pressure_loss = 0.5 * (friendly_pressure_loss + enemy_pressure_loss)
                loss = attack_loss + pressure_loss

            scaled_loss = loss / float(self.grad_accum_steps)
            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = (batch_idx % self.grad_accum_steps == 0) or (batch_idx == total_batches)
            if should_step:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item())
            attack_loss_total += float(attack_loss.item())
            pressure_loss_total += float(pressure_loss.item())
            batch_count += 1

            if progress_callback is not None and (batch_idx == total_batches or batch_idx % 10 == 0):
                elapsed_s = time.perf_counter() - started
                safe_batches = max(1, batch_count)
                safe_elapsed = max(elapsed_s, 1e-9)
                progress_callback(
                    {
                        "stage": "alphafold_training_batches",
                        "completed_batches": batch_idx,
                        "total_batches": total_batches,
                        "running_loss": total_loss / safe_batches,
                        "running_attack_loss": attack_loss_total / safe_batches,
                        "running_pressure_loss": pressure_loss_total / safe_batches,
                        "elapsed_s": elapsed_s,
                        "avg_batch_time_ms": (elapsed_s / safe_batches) * 1000.0,
                        "batches_per_s": batch_count / safe_elapsed,
                        "samples_per_s": samples_seen / safe_elapsed,
                        "samples_seen": samples_seen,
                    }
                )

        if batch_count <= 0:
            return {
                "loss": 0.0,
                "attack_loss": 0.0,
                "pressure_loss": 0.0,
                "batches": 0,
                "samples_seen": 0,
            }

        elapsed_s = time.perf_counter() - started
        safe_elapsed = max(elapsed_s, 1e-9)
        return {
            "loss": total_loss / batch_count,
            "attack_loss": attack_loss_total / batch_count,
            "pressure_loss": pressure_loss_total / batch_count,
            "batches": batch_count,
            "samples_seen": samples_seen,
            "duration_s": elapsed_s,
            "batches_per_s": batch_count / safe_elapsed,
            "samples_per_s": samples_seen / safe_elapsed,
            "avg_batch_time_ms": (elapsed_s / batch_count) * 1000.0,
        }
