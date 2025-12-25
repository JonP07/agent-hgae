from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
# import sys
# sys.path.append('/projects/standard/mhong/peng0504/hierarchy-agent/verl-agent')  # for smoke test
from verl import DataProto
import numpy as np
import torch


def _as_torch_1d(x, *, device, dtype) -> torch.Tensor:
    """Convert numpy/object-list/scalar to 1D torch tensor on device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype).view(-1)
    arr = np.asarray(x)
    if arr.dtype == object:
        # common in verl-agent: object arrays coming from torch_to_numpy(..., is_object=True)
        if dtype == torch.bool:
            arr = arr.astype(np.bool_)
        else:
            arr = arr.astype(np.float32)
    return torch.as_tensor(arr, device=device, dtype=dtype).view(-1)


def _group_indices_by_traj(traj_uid: np.ndarray) -> Dict[object, List[int]]:
    groups: Dict[object, List[int]] = defaultdict(list)
    for i, tid in enumerate(traj_uid):
        groups[tid].append(i)
    return groups


def _find_subsequence(hay: torch.Tensor, needle: Sequence[int], start: int = 0) -> int:
    """Return first position p >= start where hay[p:p+len(needle)] == needle, else -1.
    hay: 1D LongTensor (length L)
    needle: list of ints
    """
    if len(needle) == 0:
        return -1
    L = hay.numel()
    m = len(needle)
    if start < 0:
        start = 0
    if start + m > L:
        return -1
    # naive scan is fine (responses are short-ish); optimize later if needed
    needle_t = hay.new_tensor(list(needle))
    for p in range(start, L - m + 1):
        if torch.equal(hay[p : p + m], needle_t):
            return p
    return -1


def build_block_mask(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    open_ids: Sequence[int],
    close_ids: Sequence[int],
    *,
    include_tags: bool = False,
) -> torch.Tensor:
    """
    Build (N, L) boolean mask for tokens inside a single <block>...</block> span in each response.
    If tags are missing or malformed in a sample, returns all-false mask for that sample.
    """
    assert responses.ndim == 2
    N, L = responses.shape
    out = torch.zeros((N, L), device=responses.device, dtype=torch.bool)

    for i in range(N):
        # Only scan valid tokens
        valid_len = int(response_mask[i].sum().item())
        if valid_len <= 0:
            continue
        seq = responses[i, :valid_len]

        p_open = _find_subsequence(seq, open_ids, start=0)
        if p_open < 0:
            continue
        p_after_open = p_open + len(open_ids)

        p_close = _find_subsequence(seq, close_ids, start=p_after_open)
        if p_close < 0:
            continue

        if include_tags:
            s = p_open
            e = p_close + len(close_ids)
        else:
            s = p_after_open
            e = p_close

        if s < e:
            out[i, s:e] = True
    return out


@dataclass
class HGAEConfig:
    gamma: float
    lam_turn: float
    lam_seg: float
    # where to read low/high values from
    value_key_low: str = "value_low"
    value_key_high: Optional[str] = "value_high"  # if None or missing, reuse low
    # assign high-level advantage to which tokens
    assign_high_to_switch: bool = False
    assign_high_to_subgoal: bool = True
    include_tags_mask: bool = True
    norm_adv: bool = False  # whether to normalize advantages before masking
    bootstrap_truncated: bool = True  # if episode truncates, bootstrap from current value


@torch.no_grad()
def compute_hgae_advantage(
    batch,
    cfg: HGAEConfig,
    *,
    action_mask: Optional[torch.Tensor] = None,   # (N, L) bool
    subgoal_mask: Optional[torch.Tensor] = None,  # (N, L) bool
    switch_mask: Optional[torch.Tensor] = None,   # (N, L) bool
) -> DataProto:
    device = batch.batch["response_mask"].device
    response_mask = batch.batch["response_mask"].to(torch.bool)  # (N, L)
    responses = batch.batch["responses"]  # (N, L)

    N, L = responses.shape

    traj_uid = batch.non_tensor_batch["traj_uid"]
    turn_idx = np.asarray(batch.non_tensor_batch["turn_idx"]).astype(np.int64)
    dones = np.asarray(batch.non_tensor_batch["dones"]).astype(np.bool_)
    switch = np.asarray(batch.non_tensor_batch["switch"]).astype(np.bool_)

    r = _as_torch_1d(batch.non_tensor_batch["rewards"], device=device, dtype=torch.float32)  # (N,)

    # ---- helper: robustly pick value at masked position, fallback to last valid token if mask is empty ----
    def pick_value(v_tok: torch.Tensor, v_mask: torch.Tensor) -> torch.Tensor:
        v_mask = (v_mask.to(device=device, dtype=torch.bool) & response_mask)
        # fallback position: last valid token in response (or 0)
        last_pos = (response_mask.long().sum(dim=1) - 1).clamp_min(0)
        has = v_mask.any(dim=1)
        pos = torch.where(has, v_mask.float().argmax(dim=1), last_pos)  # (N,)
        return v_tok[torch.arange(N, device=device), pos]

    # ---- values (per-turn scalars) ----
    v_low_tok = batch.batch[cfg.value_key_low]             # (N, L)
    v_low_mask = batch.batch["value_mask_low"]             # (N, L) bool
    v_low = pick_value(v_low_tok, v_low_mask)              # (N,)

    if cfg.value_key_high is not None and cfg.value_key_high in batch.batch:
        v_high_tok = batch.batch[cfg.value_key_high]       # (N, L)
        v_high_mask = batch.batch["value_mask_high"]       # (N, L) bool
        v_high = pick_value(v_high_tok, v_high_mask)       # (N,)
    else:
        v_high = v_low  # single-head fallback

    # ---- default token masks ----
    if action_mask is None:
        action_mask = response_mask.clone()
    else:
        action_mask = action_mask.to(device=device, dtype=torch.bool) & response_mask

    if subgoal_mask is None:
        subgoal_mask = torch.zeros_like(response_mask, dtype=torch.bool, device=device)
    else:
        subgoal_mask = subgoal_mask.to(device=device, dtype=torch.bool) & response_mask

    if switch_mask is None:
        switch_mask = torch.zeros_like(response_mask, dtype=torch.bool, device=device)
    else:
        switch_mask = switch_mask.to(device=device, dtype=torch.bool) & response_mask

    groups = _group_indices_by_traj(traj_uid)
    zero = torch.tensor(0.0, device=device)

    # ============================================================
    # 0) Build segments (per trajectory) using boundary turns
    # boundary = start of a new subgoal segment (first turn always boundary)
    # ============================================================
    # We'll store per-traj segments as tuples (start_pos, end_pos, next_start_pos_or_None)
    traj_segments: Dict[object, List[Tuple[int, int, Optional[int]]]] = {}

    for tid, idxs in groups.items():
        idxs = sorted(idxs, key=lambda i: turn_idx[i])
        T = len(idxs)
        if T == 0:
            traj_segments[tid] = []
            continue

        boundary_pos = [0]
        for pos in range(1, T):
            if bool(switch[idxs[pos]]):
                boundary_pos.append(pos)
        boundary_pos = sorted(set(boundary_pos))

        segs = []
        for k, s_pos in enumerate(boundary_pos):
            e_pos = (boundary_pos[k + 1] - 1) if (k + 1 < len(boundary_pos)) else (T - 1)
            nxt = boundary_pos[k + 1] if (k + 1 < len(boundary_pos)) else None
            segs.append((s_pos, e_pos, nxt))
        traj_segments[tid] = segs

    # ============================================================
    # 1) Low-level (turn) GAE WITHIN each segment only.
    #    At the last turn of the segment: bootstrap to next V_high (SMDP boundary state).
    # ============================================================
    adv_low_step = torch.zeros((N,), device=device, dtype=torch.float32)
    ret_low_step = torch.zeros((N,), device=device, dtype=torch.float32)

    for tid, idxs in groups.items():
        idxs = sorted(idxs, key=lambda i: turn_idx[i])
        segs = traj_segments[tid]
        if len(idxs) == 0:
            continue

        for (s_pos, e_pos, nxt_pos) in segs:
            next_adv = zero

            # precompute bootstrap high value for this segment end
            if nxt_pos is not None:
                next_boundary_i = idxs[nxt_pos]
                boot_high = v_high[next_boundary_i].detach()
            else:
                boot_high = None

            for pos in range(e_pos, s_pos - 1, -1):
                i = idxs[pos]
                done_i = bool(dones[i])
                not_done = 0.0 if done_i else 1.0

                if pos < e_pos:
                    v_next = v_low[idxs[pos + 1]]  # within segment
                else:
                    # segment terminates here: bootstrap to next segment's high value (if episode continues)
                    if not_done > 0:
                        if boot_high is not None:
                            v_next = boot_high
                        elif cfg.bootstrap_truncated:
                            v_next = v_low[i].detach()
                        else:
                            v_next = zero
                    else:
                        v_next = zero

                delta = r[i] + cfg.gamma * not_done * v_next - v_low[i]
                adv_i = delta + cfg.gamma * cfg.lam_turn * not_done * next_adv

                adv_low_step[i] = adv_i
                ret_low_step[i] = adv_i + v_low[i]

                next_adv = adv_i

    # ============================================================
    # 2) High-level (segment) GAE: SMDP across boundaries over the whole episode
    #    delta_k = R_seg + gamma^d * V_high(next boundary) - V_high(curr boundary)
    # ============================================================
    adv_high_seg = torch.zeros((N,), device=device, dtype=torch.float32)
    ret_high_seg = torch.zeros((N,), device=device, dtype=torch.float32)

    for tid, idxs in groups.items():
        idxs = sorted(idxs, key=lambda i: turn_idx[i])
        segs = traj_segments[tid]
        if len(segs) == 0:
            continue

        next_seg_adv = zero

        for k in range(len(segs) - 1, -1, -1):
            s_pos, e_pos, nxt_pos = segs[k]
            start_i = idxs[s_pos]
            end_i = idxs[e_pos]

            # discounted segment return from s_pos..e_pos (relative discount inside segment)
            seg_return = torch.tensor(0.0, device=device)
            disc = 1.0
            for pos in range(s_pos, e_pos + 1):
                seg_return = seg_return + disc * r[idxs[pos]]
                disc *= cfg.gamma

            d_k = (e_pos - s_pos + 1)

            done_end = bool(dones[end_i])
            not_done_end = 0.0 if done_end else 1.0

            if not_done_end > 0:
                if nxt_pos is not None:
                    next_start_i = idxs[nxt_pos]
                    boot_v = v_high[next_start_i]
                elif cfg.bootstrap_truncated:
                    boot_v = v_high[start_i].detach()
                else:
                    boot_v = zero
            else:
                boot_v = zero

            target = seg_return + (cfg.gamma ** d_k) * not_done_end * boot_v
            delta_k = target - v_high[start_i]

            seg_adv_k = delta_k + (cfg.gamma ** d_k) * cfg.lam_seg * not_done_end * next_seg_adv

            adv_high_seg[start_i] = seg_adv_k
            ret_high_seg[start_i] = seg_adv_k + v_high[start_i]

            next_seg_adv = seg_adv_k

    # ============================================================
    # 3) Token assignment + masks
    # ============================================================
    # boundary turns = segment starts
    boundary_flag = torch.zeros((N,), device=device, dtype=torch.bool)
    for tid, idxs in groups.items():
        idxs = sorted(idxs, key=lambda i: turn_idx[i])
        if len(idxs) == 0:
            continue
        # always boundary at first turn
        boundary_flag[idxs[0]] = True
        # boundary at switch turns (start of new segment)
        for j in range(1, len(idxs)):
            if bool(switch[idxs[j]]):
                boundary_flag[idxs[j]] = True

    boundary_token_mask = boundary_flag.unsqueeze(-1)  # (N,1)

    # high-level advantage goes to switch/subgoal tokens ONLY on boundary turns
    hi_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    if cfg.assign_high_to_subgoal:
        hi_mask |= (subgoal_mask & boundary_token_mask)
    if cfg.assign_high_to_switch:
        hi_mask |= (switch_mask & boundary_token_mask)

    # low-level advantage goes to action tokens (you control action_mask)
    lo_mask = action_mask

    # optionally normalize advantages before masking
    if cfg.norm_adv:
        lo_turn_mask = lo_mask.any(dim=1)
        if lo_turn_mask.any():
            adv_low_mean = adv_low_step[lo_turn_mask].mean()
            adv_low_std = adv_low_step[lo_turn_mask].std(unbiased=False) + 1e-8
            adv_low_step = (adv_low_step - adv_low_mean) / adv_low_std

        hi_turn_mask = hi_mask.any(dim=1)
        if hi_turn_mask.any():
            adv_high_mean = adv_high_seg[hi_turn_mask].mean()
            adv_high_std = adv_high_seg[hi_turn_mask].std(unbiased=False) + 1e-8
            adv_high_seg = (adv_high_seg - adv_high_mean) / adv_high_std

    advantages_low = adv_low_step.unsqueeze(-1) * lo_mask.to(torch.float32)    # (N,L)
    advantages_high = adv_high_seg.unsqueeze(-1) * hi_mask.to(torch.float32)  # (N,L)

    # returns should align with critic masks (value_mask_low/high)
    vmask_low = (batch.batch["value_mask_low"].to(device=device, dtype=torch.bool) & response_mask)
    returns_low = ret_low_step.unsqueeze(-1) * vmask_low.to(torch.float32)

    if "value_mask_high" in batch.batch:
        vmask_high = (batch.batch["value_mask_high"].to(device=device, dtype=torch.bool) & response_mask)
        returns_high = ret_high_seg.unsqueeze(-1) * vmask_high.to(torch.float32)
    else:
        vmask_high = torch.zeros_like(response_mask, dtype=torch.bool)
        returns_high = torch.zeros_like(response_mask, dtype=torch.float32)

    # store outputs
    batch.batch["advantages_low"] = advantages_low
    batch.batch["returns_low"] = returns_low
    batch.batch["advantages_high"] = advantages_high
    batch.batch["returns_high"] = returns_high
    batch.batch["hgae_lo_mask"] = lo_mask
    batch.batch["hgae_hi_mask"] = hi_mask


    batch.batch["advantages"] = advantages_low + advantages_high


    return batch



def compute_value_mask(
        batch,
        tokenizer) -> torch.Tensor:
    """
    Build value mask for hierarchical GAE, to indicate which tokens should have value predictions.
    For now, we want one (or two at boundary) values per turn.
    Expects:
      batch.non_tensor_batch: traj_uid, turn_idx, switch
      batch.batch: responses, response_mask
    Produces:
      value_mask_high: (N, L) boolean tensor
      value_mask_low: (N, L) boolean tensor
    """
    device = batch.batch["response_mask"].device
    response_mask = batch.batch["response_mask"].to(torch.bool)  # (N, L)
    responses = batch.batch["responses"]  # (N, L), LongTensor

    N, L = responses.shape

    # token-id patterns for tags (plain text tags, not special tokens)
    def ids(s: str) -> torch.Tensor:
        return torch.tensor(tokenizer.encode(s, add_special_tokens=False), device=device, dtype=torch.long)

    # --- per-step signals (1D, length N) ---
    traj_uid = batch.non_tensor_batch["traj_uid"]                     # numpy object array
    turn_idx = np.asarray(batch.non_tensor_batch["turn_idx"]).astype(np.int64)
    switch = np.asarray(batch.non_tensor_batch["switch"]).astype(np.bool_)  # boundary marker

    value_mask_high = torch.zeros((N, L), device=device, dtype=torch.bool)
    value_mask_low = torch.zeros((N, L), device=device, dtype=torch.bool)

    groups = _group_indices_by_traj(traj_uid)

    for tid, idxs in groups.items():
        idxs = sorted(idxs, key=lambda i: turn_idx[i])
        T = len(idxs)

        for t in range(T):
            i = idxs[t]
            # if this turn is a boundary turn, assign two value positions: high-level value at first response token, and low-level value at the first token after </subgoal> (if exists, if not, first token after </subgoal>\n)
            # import pdb; pdb.set_trace()
            if t == 0 or bool(switch[i]):
                # first token
                if response_mask[i, 0]:
                    value_mask_high[i, 0] = True
                # find </subgoal>
                close_ids = ids("</subgoal>")
                p_close = _find_subsequence(
                    responses[i, : response_mask[i].long().sum().item()], close_ids, start=0
                )
                if p_close >= 0:
                    p_after_close = p_close + len(close_ids)
                    if p_after_close < L and response_mask[i, p_after_close]:
                        value_mask_low[i, p_after_close] = True
                else:
                    # try </subgoal>\n
                    close_ids = ids("</subgoal>\n")
                    p_close = _find_subsequence(
                        responses[i, : response_mask[i].long().sum().item()], close_ids, start=0
                    )
                    if p_close >= 0:
                        p_after_close = p_close + len(close_ids)
                        if p_after_close < L and response_mask[i, p_after_close]:
                            value_mask_low[i, p_after_close] = True
                    else:
                        # no </subgoal> found, assign low-level value after </switch>
                        close_ids = ids("</switch>")
                        p_close = _find_subsequence(
                            responses[i, : response_mask[i].long().sum().item()], close_ids, start=0
                        )
                        if p_close >= 0:
                            p_after_close = p_close + len(close_ids)
                            if p_after_close < L and response_mask[i, p_after_close]:
                                value_mask_low[i, p_after_close] = True
                        else:
                            # no </switch> either, assign low-level value after </switch>\n
                            close_ids = ids("</switch>\n")
                            p_close = _find_subsequence(
                                responses[i, : response_mask[i].long().sum().item()], close_ids, start=0
                            )
                            if p_close >= 0:
                                p_after_close = p_close + len(close_ids)
                                if p_after_close < L and response_mask[i, p_after_close]:
                                    value_mask_low[i, p_after_close] = True
                            else:
                                # no </subgoal> or </switch> found, fallback to first token as low-level value
                                value_mask_low[i, 0] = True
                # import pdb; pdb.set_trace()
            else:
                # non-boundary turn: only first token after </subgoal> or </subgoal>\n
                close_ids = ids("</subgoal>")
                p_close = _find_subsequence(
                    responses[i, : response_mask[i].long().sum().item()], close_ids, start=0
                )
                if p_close >= 0:
                    p_after_close = p_close + len(close_ids)
                    if p_after_close < L and response_mask[i, p_after_close]:
                        value_mask_low[i, p_after_close] = True
                else:
                    # try </subgoal>\n
                    close_ids = ids("</subgoal>\n")
                    p_close = _find_subsequence(
                        responses[i, : response_mask[i].long().sum().item()], close_ids, start=0
                    )
                    if p_close >= 0:
                        p_after_close = p_close + len(close_ids)
                        if p_after_close < L and response_mask[i, p_after_close]:
                            value_mask_low[i, p_after_close] = True
                    else:
                        # no </subgoal> found, assign low-level value after </switch>
                        close_ids = ids("</switch>")
                        p_close = _find_subsequence(
                            responses[i, : response_mask[i].long().sum().item()], close_ids, start=0
                        )
                        if p_close >= 0:
                            p_after_close = p_close + len(close_ids)
                            if p_after_close < L and response_mask[i, p_after_close]:
                                value_mask_low[i, p_after_close] = True
                        else:
                            # no </switch> either, assign low-level value after </switch>\n
                            close_ids = ids("</switch>\n")
                            p_close = _find_subsequence(
                                responses[i, : response_mask[i].long().sum().item()], close_ids, start=0
                            )
                            if p_close >= 0:
                                p_after_close = p_close + len(close_ids)
                                if p_after_close < L and response_mask[i, p_after_close]:
                                    value_mask_low[i, p_after_close] = True
                            else:
                                # no </subgoal> or </switch> found, fallback to first token as low-level value
                                value_mask_low[i, 0] = True
                # pdb.set_trace()
    return value_mask_high, value_mask_low

if __name__ == "__main__":
    from verl.utils import hf_processor, hf_tokenizer
    tokenizer = hf_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    print("Tokenizer loaded.")
    test_response = "<switch>SWITCH</switch>\n<subgoal>Do something</subgoal>\n<action>Now continue</action>"
    def ids(s: str) -> torch.Tensor:
        return torch.tensor(tokenizer.encode(s, add_special_tokens=False), device='cpu', dtype=torch.long)
    
    resp_ids = ids(test_response)  # (1, L)
    print("Response IDs:", resp_ids)
    p_close = _find_subsequence(resp_ids[0], ids("</subgoal>"), start=0)
    print("Position of </subgoal>:", p_close)

    # try </subgoal>\n
    p_close = _find_subsequence(resp_ids[0], ids("</subgoal>\n"), start=0)
    print("Position of </subgoal>\\n:", p_close)

    print("</subgoal> IDs:", ids("</subgoal>"))
    print("</subgoal>\n IDs:", ids("</subgoal>\n"))
