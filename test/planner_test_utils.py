"""
Utility helpers for planner tests.
"""

from __future__ import annotations

import uuid

import torch

from setu._commons.datatypes import (
    CopySpec,
    Device,
    TensorDim,
    TensorSelection,
    TensorShardSpec,
    TensorSlice,
)
from setu._metastore import MetaStore
from setu._planner import NCCLPlanner


class PlannerTestHelper:
    """Convenience wrapper around NCCLPlanner + MetaStore for tests."""

    def __init__(self):
        self.planner = NCCLPlanner()
        self.metastore = MetaStore()
        self._selections = {}
        self.name_to_id = {}
        self.id_to_name = {}
        self.shard_id_to_name = {}

    def register_tensor(
        self,
        tensor_name,
        shard_name,
        device,
        node_name,
        dims,
        dtype=torch.float32,
    ):
        """Register a tensor shard and store its full-tensor selection."""
        if node_name in self.name_to_id:
            node_id = self.name_to_id[node_name]
        else:
            node_id = uuid.uuid4()
            self.name_to_id[node_name] = node_id
            self.id_to_name[node_id] = node_name

        shard_spec = TensorShardSpec(tensor_name, dims, dtype, Device(device))
        metadata = self.metastore.register_tensor_shard(shard_spec, node_id)
        self.shard_id_to_name[metadata.id] = shard_name

        dim_map = {d.name: TensorDim(d.name, d.size) for d in dims}
        self._selections[tensor_name] = TensorSelection(tensor_name, dim_map)

    def select(self, tensor_name, slices=None):
        """Create a TensorSelection, optionally narrowed by slices."""
        sel = self._selections[tensor_name]
        if slices is not None:
            for dim_name, (start, end) in slices.items():
                sel = sel.where(dim_name, TensorSlice(dim_name, start, end))
        return sel

    def compile(self, src, dst):
        """Build a CopySpec and compile the plan."""
        copy_spec = CopySpec(src.name, dst.name, src, dst)
        return self.planner.compile(copy_spec, self.metastore)

    def _format_participant(self, part):
        node_name = self.id_to_name.get(part.node_id, str(part.node_id))
        return f"Participant(node={node_name}, device={part.device})"

    def _format_shard(self, shard_ref):
        shard_name = self.shard_id_to_name.get(
            shard_ref.shard_id, str(shard_ref.shard_id)
        )
        node_name = self.id_to_name.get(shard_ref.node_id, str(shard_ref.node_id))
        return f"({shard_name}, node={node_name})"

    def _sorted_ranks(self, participant_to_rank):
        """Return (participant, rank) pairs sorted by formatted participant name."""
        return sorted(
            participant_to_rank.items(),
            key=lambda kv: self._format_participant(kv[0]),
        )

    def _build_rank_remap(self, plan, sorted_parts):
        """Build old_rank -> new_rank mapping per communicator.

        Assigns normalized ranks 0, 1, 2, ... in alphabetical participant order
        so that output is deterministic regardless of UUID ordering.
        """
        remap = {}
        for part in sorted_parts:
            for instr in plan.program[part]:
                if instr.type_name != "InitComm":
                    continue
                key = repr(instr.inner.comm_id)
                if key in remap:
                    continue
                remap[key] = {
                    old_rank: new_rank
                    for new_rank, (_, old_rank) in enumerate(
                        self._sorted_ranks(instr.inner.participant_to_rank)
                    )
                }
        return remap

    def format_plan(self, plan):
        """Format plan into a deterministic string."""
        comm_counter = 0
        comm_map = {}

        def rename_comm(comm_id):
            nonlocal comm_counter
            key = repr(comm_id)
            if key not in comm_map:
                comm_map[key] = f"COMM_{comm_counter}"
                comm_counter += 1
            return comm_map[key]

        sorted_parts = sorted(
            plan.program.keys(),
            key=lambda p: (self.id_to_name.get(p.node_id, ""), str(p.device)),
        )
        rank_remap = self._build_rank_remap(plan, sorted_parts)

        lines = [f"Plan(num_participants={len(plan.participants)}):"]

        for part in sorted_parts:
            lines.append(f"  {self._format_participant(part)}:")
            current_comm_key = None

            for instr in plan.program[part]:
                name = instr.type_name
                inner = instr.inner

                if name in ("InitComm", "UseComm"):
                    current_comm_key = repr(inner.comm_id)

                if name == "InitComm":
                    remap = rank_remap[current_comm_key]
                    ranks = {
                        self._format_participant(p): remap[r]
                        for p, r in self._sorted_ranks(inner.participant_to_rank)
                    }
                    lines.append(
                        f"    InitComm(comm={rename_comm(inner.comm_id)}, ranks={ranks})"
                    )
                elif name == "UseComm":
                    lines.append(f"    UseComm(comm={rename_comm(inner.comm_id)})")
                elif name == "Copy":
                    lines.append(
                        f"    Copy(src={self._format_shard(inner.src_shard)}, "
                        f"src_offset={inner.src_offset_bytes}, "
                        f"dst={self._format_shard(inner.dst_shard)}, "
                        f"dst_offset={inner.dst_offset_bytes}, "
                        f"count={inner.count}, dtype={inner.dtype})"
                    )
                elif name == "Send":
                    peer = rank_remap[current_comm_key][inner.peer_rank]
                    lines.append(
                        f"    Send(src={self._format_shard(inner.src_shard)}, "
                        f"offset={inner.offset_bytes}, count={inner.count}, "
                        f"dtype={inner.dtype}, peer_rank={peer})"
                    )
                elif name == "Receive":
                    peer = rank_remap[current_comm_key][inner.peer_rank]
                    lines.append(
                        f"    Receive(dst={self._format_shard(inner.dst_shard)}, "
                        f"offset={inner.offset_bytes}, count={inner.count}, "
                        f"dtype={inner.dtype}, peer_rank={peer})"
                    )
                else:
                    raise ValueError(f"Unknown instruction type: {name}")

        return "\n".join(lines)
