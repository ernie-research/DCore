# coding=utf-8
"""DSpark composite loss: CE + L1 distillation + confidence BCE."""

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_ce_loss(
    draft_logits: torch.Tensor,
    target_ids: torch.Tensor,
    weight_mask: torch.Tensor,
    valid_token_count: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss against ground-truth token ids.

    Args:
        draft_logits: (*, V) unnormalized logits from draft model.
        target_ids: (*) ground-truth token ids.
        weight_mask: (*) per-token weight (0 for masked positions).
        valid_token_count: scalar denominator (pre-reduced across ranks).

    Returns:
        Scalar weighted CE loss.
    """
    flat_logits = draft_logits.reshape(-1, draft_logits.size(-1))
    flat_targets = target_ids.view(-1)
    flat_weights = weight_mask.view(-1)
    ce_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    return (ce_per_token * flat_weights).sum() / valid_token_count


def compute_l1_distillation_loss(
    draft_logits: torch.Tensor,
    target_probs: torch.Tensor,
    weight_mask: torch.Tensor,
    valid_token_count: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """L1 distance between draft and target probability distributions (= 2*TV).

    Args:
        draft_logits: (..., V) unnormalized logits from draft model.
        target_probs: (..., V) target probability distribution (detached).
        weight_mask: (...) per-token weight.
        valid_token_count: scalar denominator.

    Returns:
        (l1_loss, l1_per_token): scalar loss and per-token L1 values.
    """
    draft_probs = F.softmax(draft_logits.float(), dim=-1)
    l1_per_token = (draft_probs - target_probs).abs().sum(dim=-1)
    l1_loss = (l1_per_token.to(weight_mask.dtype) * weight_mask).sum() / valid_token_count
    return l1_loss, l1_per_token


def compute_confidence_loss(
    conf_logits: torch.Tensor,
    l1_per_token: torch.Tensor,
    weight_mask: torch.Tensor,
    valid_token_count: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy for confidence head predicting acceptance probability.

    The acceptance target is always the soft label derived from the L1 distance
    to the target distribution:
        target = clamp(1 - 0.5 * l1_per_token, 0, 1)   (= 1 - TV)
    A hard-label fallback ("did argmax hit the ground truth") would silently
    change what the head learns whenever the distillation alphas are 0, so the
    caller must supply l1_per_token.

    Args:
        conf_logits: (...) raw confidence predictions (pre-sigmoid).
        l1_per_token: (...) per-token L1 between draft and target distributions.
        weight_mask: (...) per-token weight.
        valid_token_count: scalar denominator.

    Returns:
        Scalar confidence BCE loss.
    """
    if l1_per_token is None:
        raise ValueError(
            "compute_confidence_loss requires l1_per_token; the acceptance "
            "target must be the soft 1 - 0.5 * L1 label."
        )
    with torch.no_grad():
        acceptance_target = (1.0 - 0.5 * l1_per_token).clamp(0, 1)

    return F.binary_cross_entropy_with_logits(
        conf_logits.float(), acceptance_target, weight=weight_mask, reduction="sum"
    ) / valid_token_count


def compute_dspark_loss(
    *,
    draft_logits: torch.Tensor,
    target_ids: torch.Tensor,
    target_probs: Optional[torch.Tensor] = None,
    conf_logits: Optional[torch.Tensor] = None,
    weight_mask: torch.Tensor,
    valid_token_count: torch.Tensor,
    ce_alpha: float = 0.1,
    l1_alpha: float = 0.9,
    confidence_alpha: float = 1.0,
    world_size: int = 1,
) -> Tuple[torch.Tensor, dict]:
    """Compute the full DSpark composite loss: CE + L1 + Confidence.

    Args:
        draft_logits: (..., V) draft model logits.
        target_ids: (...) ground-truth token ids.
        target_probs: (..., V) target distribution (detached). If None, L1 is skipped.
        conf_logits: (...) confidence head logits. If None, confidence loss is skipped.
        weight_mask: (...) per-token weight mask.
        valid_token_count: scalar denominator (pre-reduced across ranks).
        ce_alpha: weight for CE loss.
        l1_alpha: weight for L1 distillation loss.
        confidence_alpha: weight for confidence BCE loss.
        world_size: number of distributed ranks (loss is scaled by this).

    Returns:
        (loss, metrics): total scalar loss and dict of per-component detached values.
    """
    device = draft_logits.device

    ce_loss = compute_ce_loss(draft_logits, target_ids, weight_mask, valid_token_count)

    l1_loss = torch.tensor(0.0, device=device)
    l1_per_token = None
    if target_probs is not None:
        l1_loss, l1_per_token = compute_l1_distillation_loss(
            draft_logits, target_probs, weight_mask, valid_token_count
        )

    conf_loss = torch.tensor(0.0, device=device)
    if confidence_alpha > 0 and conf_logits is not None:
        conf_loss = compute_confidence_loss(
            l1_per_token=l1_per_token,
            conf_logits=conf_logits,
            weight_mask=weight_mask,
            valid_token_count=valid_token_count,
        )

    loss = (ce_alpha * ce_loss + l1_alpha * l1_loss + confidence_alpha * conf_loss) * world_size

    metrics = {
        "ce_loss": ce_loss.detach(),
        "l1_loss": l1_loss.detach(),
        "conf_loss": conf_loss.detach(),
    }
    return loss, metrics
