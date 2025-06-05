from dataclasses import dataclass
from typing import Literal, Optional

import einops
import torch

import time

import numpy as np
import pulp
from typing import List
from sglang.srt.managers.schedule_batch import (
    get_global_expert_location_metadata,
    global_server_args_dict,
)
from sglang.srt.utils import get_compiler_backend


@dataclass
class ExpertLocationDispatchInfo:
    ep_dispatch_algorithm: Literal["static", "random", "workload_based"]
    partial_logical_to_rank_dispatch_physical_map: torch.Tensor
    partial_logical_to_all_physical_map: torch.Tensor
    partial_logical_to_all_physical_map_num_valid: torch.Tensor
    num_physical_experts: int

    @classmethod
    def init_new(cls, ep_rank: int, layer_id: int):
        ep_dispatch_algorithm = global_server_args_dict["ep_dispatch_algorithm"]
        expert_location_metadata = get_global_expert_location_metadata()

        if ep_dispatch_algorithm is None:
            return None

        return cls(
            ep_dispatch_algorithm=ep_dispatch_algorithm,
            partial_logical_to_rank_dispatch_physical_map=expert_location_metadata.logical_to_rank_dispatch_physical_map[
                ep_rank, layer_id, :
            ],
            partial_logical_to_all_physical_map=expert_location_metadata.logical_to_all_physical_map[
                layer_id, :
            ],
            partial_logical_to_all_physical_map_num_valid=expert_location_metadata.logical_to_all_physical_map_num_valid[
                layer_id, :
            ],
            num_physical_experts=expert_location_metadata.num_physical_experts,
        )


def topk_ids_logical_to_physical(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    
    if info is None:
        return topk_ids

    if info.ep_dispatch_algorithm == "static":
        return _topk_ids_logical_to_physical_static(topk_ids, info)
    if info.ep_dispatch_algorithm == "random":
        return _topk_ids_logical_to_physical_random(topk_ids, info)
    if info.ep_dispatch_algorithm == "fake_uniform":
        return _topk_ids_logical_to_physical_fake_uniform(topk_ids, info)
    if info.ep_dispatch_algorithm == "fake_grouped_uniform":
        return _topk_ids_logical_to_physical_fake_grouped_uniform(topk_ids, info)
    if info.ep_dispatch_algorithm == "workload_based":
        return _topk_ids_logical_to_physical_workload_based(topk_ids, info)
    raise NotImplementedError


def _topk_ids_logical_to_physical_static(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    return info.partial_logical_to_rank_dispatch_physical_map[topk_ids]


def _topk_ids_logical_to_physical_random(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    topk_ids_original_shape = topk_ids.shape
    device = topk_ids.device
    topk_ids = topk_ids.flatten()

    chosen_dispatch_index = (
        torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device)
        % info.partial_logical_to_all_physical_map_num_valid[topk_ids]
    )
    topk_ids = info.partial_logical_to_all_physical_map[topk_ids, chosen_dispatch_index]

    topk_ids = topk_ids.view(topk_ids_original_shape)
    return topk_ids


def _topk_ids_logical_to_physical_workload_based(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    print("~~~~~~~~workload!!!!!")
    device = topk_ids.device
    E = info.partial_logical_to_all_physical_map.size(0)
    flat_logical = topk_ids.flatten().to(torch.int64)

    # workload vector  
    workload_np = _logical_workload(topk_ids, E)           # (E,1)

    # presence matrix  x_mat[e,g]
    x_mat_np, G, M  = _presence_matrix(info)                   # (E,G)

    start_time = time.time()
    # run solver  ->  y[e,g] tokens per GPU 
    y_np = _solve_token_routing_pulp(workload_np, x_mat_np, G)
    end_time = time.time()
    print(f"MILP Solving Time: {end_time - start_time:.6f} seconds")

    phys_ids = info.partial_logical_to_all_physical_map
    num_valid = info.partial_logical_to_all_physical_map_num_valid
    lookup_phys_id = torch.full((E, G), -1, dtype=torch.int32, device=device)

    for e in range(E):
        p_valid = int(num_valid[e].item())
        for p in phys_ids[e, :p_valid].tolist():
            g = p // M
            lookup_phys_id[e, g] = p
    
    sorted_e, token_pos = torch.sort(flat_logical)  # sort token indices by logical expert id
    # Find segment boundaries where logical id changes
    seg_start = torch.cat([
        torch.tensor([0], device=device),
        torch.nonzero(sorted_e[1:] != sorted_e[:-1]).flatten() + 1,
        torch.tensor([sorted_e.numel()], device=device)
    ])
    num_seg = seg_start.numel() - 1

    # Bucket token positions for each logical expert
    token_pos_per_e = [
        token_pos[seg_start[i]: seg_start[i + 1]]
        for i in range(num_seg)
    ] # len is number of logical expert
    unique_e_ids = sorted_e[seg_start[:-1]]
    e_to_index = {e.item(): i for i, e in enumerate(unique_e_ids)}

    N = flat_logical.numel()
    flat_physical = torch.empty(N, dtype=torch.int32, device=device)

    per_expert_cursors = [0] * E
    token_chunks, phys_chunks = [], []
    # map key: expert id, value: token indices
    # [2,3,4,5,6,7,8]
    # map expert_id, gpu; number of tokens
    # token index -> physical idx
    # for each gpu, assign exact number of tokens to the expert copy according to y
    for g in range(G):
        # Which logical experts e send >=1 token to GPU g?
        col = torch.tensor(y_np[:, g], device=device)
        e_idx = torch.nonzero(col).flatten()  # 1D tensor of expert IDs

        if e_idx.numel() == 0:
            continue

        # For each such expert e, slice the next y_np[e,g] tokens and assign to phys_id
        for e in e_idx.tolist():
            quota = int(y_np[e, g])  # number of tokens of expert e → GPU g
            if quota <= 0:
                continue

            cursor = per_expert_cursors[e]
            end = cursor + quota
            per_expert_cursors[e] = end

            # Find position in token_pos_per_e using e_to_index
            idx_list = e_to_index[e]
            tok_indices = token_pos_per_e[idx_list][cursor:end]  # shape (quota,)

            # Look up the correct physical ID (int32) for expert e on GPU g
            phys_id = lookup_phys_id[e, g].item()
            assert phys_id >= 0, f"No physical replica of expert {e} on GPU {g}"

            phys_tensor = torch.full((quota,), phys_id, dtype=torch.int32, device=device)

            token_chunks.append(tok_indices)   # (quota,)
            phys_chunks.append(phys_tensor)    # (quota,)

    # 8. Concatenate all and scatter into flat_physical
    if token_chunks:
        all_tokens = torch.cat(token_chunks)  # shape (total_assigned_tokens,)
        all_phys = torch.cat(phys_chunks)     # same shape
        # all_tokens must be long for indexing
        all_tokens = all_tokens.to(dtype=torch.long)

        # Now assign
        flat_physical[all_tokens] = all_phys

    # 9. Reshape back to (N, K) if needed (here we assume input was flattened)
    return flat_physical.view_as(topk_ids)

    # # assemble final token-level mapping 
    # # Pre-compute, for each logical expert, the list of physical IDs on every GPU
    # phys_ids  = info.partial_logical_to_all_physical_map.cpu()        # [E, P_max]
    # num_valid = info.partial_logical_to_all_physical_map_num_valid    # [E]

    # # Token indices per logical expert
    # buckets: List[List[int]] = [[] for _ in range(E)]
    # flat_logical = topk_ids.flatten()
    # for idx, e_id in enumerate(flat_logical.tolist()):
    #     buckets[e_id].append(idx)

    # # Output tensor we will fill (1-D then reshape)
    # flat_physical = torch.empty_like(flat_logical, device=device)

    # # Iterate logical experts and distribute tokens according to y_np
    # print("~~~~~~~~Loop start~~~~~~")
    # for e in range(E):
    #     token_idx_list = buckets[e]
    #     if not token_idx_list:
    #         continue

    #     # Build a queue of (gpu_id, remaining_quota_on_gpu, physical_id_list_for_gpu)
    #     p_valid = int(num_valid[e])
    #     phys_list_e = phys_ids[e, :p_valid].tolist()           # ordered physical IDs

    #     # Group them by GPU according to phys_id // M
    #     gpu_to_phys: List[List[int]] = [[] for _ in range(G)]
    #     for p in phys_list_e:
    #         gpu_to_phys[p // M].append(p)

    #     # Consume quotas y_np[e, g] in round-robin order within each GPU’s list
    #     ptr_in_gpu = [0] * G
    #     for g in range(G):
    #         quota = y_np[e, g]
    #         if quota == 0 or not gpu_to_phys[g]:
    #             continue
    #         phys_ids_g = gpu_to_phys[g]
    #         sz = len(phys_ids_g)
    #         for q in range(quota):
    #             tok_global_idx = token_idx_list.pop()          # pop from end (O(1))
    #             p_id = phys_ids_g[ptr_in_gpu[g] % sz]
    #             ptr_in_gpu[g] += 1
    #             flat_physical[tok_global_idx] = p_id
    # print("~~~~~~~~`Loop ends~~~~~~~~~")
    # # There should be no tokens left unassigned
    # assert all(len(v) == 0 for v in buckets), \
    #     "Some tokens were not mapped to physical experts"

    # return flat_physical.view_as(topk_ids)


def _topk_ids_logical_to_physical_fake_uniform(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    # NOTE it will have probability to send one token to one expert multiple times
    return torch.randint(
        0,
        info.num_physical_experts,
        topk_ids.shape,
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _topk_ids_logical_to_physical_fake_grouped_uniform(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    # NOTE it will have probability to send one token to one expert multiple times
    # NOTE it will make each group have exactly two experts chosen

    num_tokens, num_topk = topk_ids.shape
    dtype = topk_ids.dtype
    device = topk_ids.device
    num_physical_experts = info.num_physical_experts

    n_group = 8
    topk_group = 4
    num_experts_per_group = num_physical_experts // n_group

    chosen_groups_of_token = torch.rand(num_tokens, n_group, device=device).argsort(
        dim=1
    )[:, :topk_group]
    delta_within_group = torch.randint(
        0,
        num_physical_experts // n_group,
        (num_tokens, num_topk),
        dtype=dtype,
        device=device,
    )
    chosen_groups_of_token_repeated = einops.repeat(
        chosen_groups_of_token,
        "num_tokens topk_group -> num_tokens (topk_group repeat_n)",
        repeat_n=num_topk // topk_group,
    )
    return chosen_groups_of_token_repeated * num_experts_per_group + delta_within_group


def _logical_workload(
    topk_ids: torch.Tensor, num_logical: int
) -> np.array:
    flat = topk_ids.flatten()
    workload = torch.bincount(flat, minlength=num_logical)
    return workload.cpu().numpy().astype(int).reshape((-1,1))


def _presence_matrix(
    info: Optional[ExpertLocationDispatchInfo]
) -> np.array:
    phys_ids: torch.Tensor = info.partial_logical_to_all_physical_map
    num_valid: torch.Tensor = info.partial_logical_to_all_physical_map_num_valid
    E, P_max = phys_ids.shape
    phys_ids_cpu = phys_ids.cpu()
    num_valid_cpu = num_valid.cpu()

    num_physical = int(info.num_physical_experts)
    default_G   = 4
    G           = default_G
    num_red = global_server_args_dict["ep_num_redundant_experts"]
    M           = (num_physical + num_red) // G
    #assert G * M == num_physical

    x_mat = np.zeros((E, G), dtype=int)
    for e in range(E):
        k = int(num_valid_cpu[e])
        for j in range(k):
            p_id = int(phys_ids_cpu[e, j])
            gpu = p_id // M
            x_mat[e, gpu] = 1
    return x_mat, G, M


def _solve_token_routing_pulp(
    workload: np.ndarray, x_mat: np.ndarray, G: int
) -> np.array:
    """
    workload : (E,1)  int
    x_mat    : (E,G)  {0,1}
    returns  : (E,G)  int   (tokens per logical expert per GPU)
    """
    E = workload.shape[0]

    prob = pulp.LpProblem("TokenRoutingFixedLayout", pulp.LpMinimize)

    # Variables
    y = [[pulp.LpVariable(f"y_{e}_{g}", lowBound=0, cat='Integer')
          for g in range(G)] for e in range(E)]
    z      = [pulp.LpVariable(f"z_{g}",   lowBound=0, cat='Integer') for g in range(G)]
    z_max  = pulp.LpVariable("z_max",     lowBound=0, cat='Integer')

    # Constraints:
    for e in range(E):
        prob += pulp.lpSum(y[e][g] for g in range(G)) == int(workload[e])
        for g in range(G):
            prob += y[e][g] <= x_mat[e, g] * int(workload[e])  # cannot assign to GPU where expert e isn't present

    for g in range(G):
        prob += z[g] == pulp.lpSum(y[e][g] for e in range(E))  # total tokens assigned to GPU g
        prob += z[g] <= z_max

    # Objective
    prob += z_max

    # Solve
    print("~~~~~~~Solver started~~~~~~~~~")
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    print("~~~~~~~Solver finished~~~~~~~")
    # Extract results
    y_np = np.array([[int(y[e][g].value()) for g in range(G)] for e in range(E)], dtype=int)  # shape: (E, G)
    return y_np