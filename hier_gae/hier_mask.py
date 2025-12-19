from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional, Dict, Any

import numpy as np
import torch


def _find_subseq_1d(hay: torch.Tensor, needle: torch.Tensor, start: int = 0) -> int:
    """Return first p >= start where hay[p:p+m] == needle, else -1. hay is 1D LongTensor."""
    m = needle.numel()
    L = hay.numel()
    if m == 0 or start + m > L:
        return -1
    for p in range(start, L - m + 1):
        if torch.equal(hay[p : p + m], needle):
            return p
    return -1


def _span_from_tags(
    seq: torch.Tensor,
    valid_len: int,
    open_ids: torch.Tensor,
    close_ids: torch.Tensor,
    *,
    include_tags: bool = False,
) -> Optional[Tuple[int, int]]:
    """Return [s,e) span in token indices within seq[:valid_len], or None if malformed/missing."""
    if valid_len <= 0:
        return None
    s_open = _find_subseq_1d(seq[:valid_len], open_ids, start=0)
    if s_open < 0:
        return None
    s_content = s_open + int(open_ids.numel())
    s_close = _find_subseq_1d(seq[:valid_len], close_ids, start=s_content)
    if s_close < 0:
        return None
    if include_tags:
        return (s_open, s_close + int(close_ids.numel()))
    return (s_content, s_close)


@dataclass
class HGMaskConfig:
    # which switch values mean "do NOT start a new subgoal segment"
    keep_values: Tuple[str, ...] = ("KEEP",)
    # if tag extraction fails, fall back to training all valid response tokens as action
    fallback_action_to_full_response: bool = True
    # if switch tag missing, treat as KEEP by default
    default_switch_value: str = "KEEP"


@torch.no_grad()
def make_hgae_masks_and_switch(
    batch,
    tokenizer,
    include_tags_mask: bool = False,
    cfg: HGMaskConfig = HGMaskConfig(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Returns:
      action_mask:  (N,L) bool
      subgoal_mask: (N,L) bool
      switch_mask:  (N,L) bool   (content inside <switch>...</switch>)
      is_new_subgoal: np.ndarray (N,) bool  True means start a new segment here
    """
    responses: torch.Tensor = batch.batch["responses"]          # (N,L) long
    response_mask: torch.Tensor = batch.batch["response_mask"]  # (N,L) 0/1
    device = responses.device
    response_mask = response_mask.to(torch.bool)

    N, L = responses.shape
    valid_lens = response_mask.long().sum(dim=1).cpu().numpy().astype(int)

    # token-id patterns for tags (plain text tags, not special tokens)
    def ids(s: str) -> torch.Tensor:
        return torch.tensor(tokenizer.encode(s, add_special_tokens=False), device=device, dtype=torch.long)

    open_switch, close_switch = ids("<switch>"), ids("</switch>")
    close_switch_alt = ids('</switch>\n')  # handle possible newline after tag
    open_subgoal, close_subgoal = ids("<subgoal>"), ids("</subgoal>")
    close_subgoal_alt = ids('</subgoal>\n')  # handle possible newline after tag
    open_action, close_action = ids("<action>"), ids("</action>")
    close_action_alt = ids('</action>\n')  # handle possible newline after tag

    action_mask = torch.zeros((N, L), device=device, dtype=torch.bool)
    subgoal_mask = torch.zeros((N, L), device=device, dtype=torch.bool)
    switch_mask = torch.zeros((N, L), device=device, dtype=torch.bool)

    is_new_subgoal = np.zeros((N,), dtype=np.bool_)

    keep_set = {k.strip().upper() for k in cfg.keep_values}

    for i in range(N):
        vl = int(valid_lens[i])
        if vl <= 0:
            continue
        seq = responses[i, :vl]

        # ---------- ACTION mask (masking uses include_tags) ----------
        sp = _span_from_tags(seq, vl, open_action, close_action, include_tags=include_tags_mask)
        sp_alt = _span_from_tags(seq, vl, open_action, close_action_alt, include_tags=include_tags_mask)
        if sp is None and sp_alt is not None:
            sp = sp_alt
        if sp is None:
            if cfg.fallback_action_to_full_response:
                action_mask[i, :vl] = True
        else:
            s, e = sp
            if s < e:
                action_mask[i, s:e] = True

        # ---------- SUBGOAL mask (masking uses include_tags) ----------
        sp = _span_from_tags(seq, vl, open_subgoal, close_subgoal, include_tags=include_tags_mask)
        sp_alt = _span_from_tags(seq, vl, open_subgoal, close_subgoal_alt, include_tags=include_tags_mask)
        if sp is None and sp_alt is not None:
            sp = sp_alt
        if sp is not None:
            s, e = sp
            if s < e:
                subgoal_mask[i, s:e] = True

        # ---------- SWITCH: (1) content-only extraction, (2) include_tags for masking ----------
        # (1) content-only span for boundary decision
        sp_content = _span_from_tags(seq, vl, open_switch, close_switch, include_tags=False)
        sp_content_alt = _span_from_tags(seq, vl, open_switch, close_switch_alt, include_tags=False)
        if sp_content is None and sp_content_alt is not None:
            sp_content = sp_content_alt

        # missing OR empty content => KEEP
        switch_text = cfg.default_switch_value.strip().upper()
        if sp_content is not None:
            s_c, e_c = sp_content
            if s_c < e_c:  # non-empty content
                txt = tokenizer.decode(seq[s_c:e_c].tolist(), skip_special_tokens=False).strip().upper()
                if txt:  # non-whitespace
                    switch_text = txt

        # (2) mask span: include_tags controls whether tags are included in switch_mask
        sp_mask = _span_from_tags(seq, vl, open_switch, close_switch, include_tags=include_tags_mask)
        sp_mask_alt = _span_from_tags(seq, vl, open_switch, close_switch_alt, include_tags=include_tags_mask)
        if sp_mask is None and sp_mask_alt is not None:
            sp_mask = sp_mask_alt
        if sp_mask is not None:
            s_m, e_m = sp_mask
            if s_m < e_m:
                switch_mask[i, s_m:e_m] = True

        is_new_subgoal[i] = (switch_text not in keep_set)

    # ensure masks only on valid response tokens
    action_mask &= response_mask
    subgoal_mask &= response_mask
    switch_mask &= response_mask

    return action_mask, subgoal_mask, switch_mask, is_new_subgoal
