# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.fx as fx
from torch.fx.experimental.symbolic_shapes import statically_known_true

from vllm.logger import init_logger

from ..fx_utils import is_func
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


def _same_shape(a: torch.Size, b: torch.Size) -> bool:
    return len(a) == len(b) and all(
        statically_known_true(x == y) for x, y in zip(a, b)
    )


def _is_foldable_add(node: object) -> bool:
    """An ``a + b`` whose only user is the all-reduce, with no alpha, no
    broadcasting and no type promotion."""
    if not (
        isinstance(node, fx.Node)
        and is_func(node, torch.ops.aten.add.Tensor)
        and len(node.args) == 2
        and not node.kwargs
        and len(node.users) == 1
    ):
        return False
    vals = [node.meta.get("val")]
    for arg in node.args:
        if not isinstance(arg, fx.Node):
            return False
        vals.append(arg.meta.get("val"))
    if not all(isinstance(v, torch.Tensor) for v in vals):
        return False
    out = vals[0]
    return all(v.dtype == out.dtype and _same_shape(v.shape, out.shape) for v in vals)


class AllReduceAddFusionPass(VllmInductorPass):
    """Fold an elementwise add into the all-reduce that consumes it:

        all_reduce(a + b) -> all_reduce_add(a, b)

    For example the shared-expert + routed-expert add in front of the MoE
    all-reduce. Backends that can't fold the add fall back to
    ``all_reduce(a + b)`` at runtime, so the rewrite is always exact.

    Must run after the all-reduce + RMSNorm fusion, which also consumes
    ``all_reduce`` and removes more work when it matches.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.matched_count = 0

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        count = 0
        for node in list(graph.nodes):
            if not is_func(node, torch.ops.vllm.all_reduce.default):
                continue
            add = node.args[0]
            if not _is_foldable_add(add):
                continue
            group_name = node.kwargs.get("group_name", None)
            if group_name is None:
                group_name = node.args[1]
            with graph.inserting_before(node):
                fused = graph.call_function(
                    torch.ops.vllm.all_reduce_add.default,
                    args=tuple(add.args),
                    kwargs={"group_name": group_name},
                )
            fused.meta.update(node.meta)
            node.replace_all_uses_with(fused)
            graph.erase_node(node)
            graph.erase_node(add)
            count += 1

        self.matched_count = count
        VllmPatternMatcherPass.match_table[self.pass_name] += count
        logger.debug("Folded %s adds into all_reduce", count)
