from typing import Sequence

import infinicore
from infinicore.tensor import Tensor


def column_stack(tensors: Sequence[Tensor], *, out=None) -> Tensor:
    if (
        infinicore.use_ntops
        and tensors[0].device.type in ("cuda", "musa")
        and out is None
    ):
        return infinicore.ntops.torch.column_stack(tensors)

    if out is None:
        return Tensor(
            _infinicore.column_stack([t._underlying for t in tensors])
        )

    _infinicore.column_stack_(
        out._underlying, [t._underlying for t in tensors]
    )
    return out
