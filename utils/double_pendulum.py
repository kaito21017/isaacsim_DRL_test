"""Shared math helpers for the double-pendulum task."""

from __future__ import annotations

import math

import torch


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi]."""
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def compute_upright_errors(
    joint1_pos: torch.Tensor,
    joint2_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-link angular error relative to the upright configuration."""
    link1_error = wrap_to_pi(joint1_pos - math.pi)
    link2_error = wrap_to_pi(joint1_pos + joint2_pos - math.pi)
    return link1_error, link2_error


def compute_upright_mask(
    joint1_pos: torch.Tensor,
    joint2_pos: torch.Tensor,
    joint1_vel: torch.Tensor,
    joint2_vel: torch.Tensor,
    *,
    angle_threshold: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Return True when both links are upright and nearly stationary."""
    link1_error, link2_error = compute_upright_errors(joint1_pos, joint2_pos)
    return (
        (torch.abs(link1_error) <= angle_threshold)
        & (torch.abs(link2_error) <= angle_threshold)
        & (torch.abs(joint1_vel) <= velocity_threshold)
        & (torch.abs(joint2_vel) <= velocity_threshold)
    )
