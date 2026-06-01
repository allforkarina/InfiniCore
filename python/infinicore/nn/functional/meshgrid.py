import infinicore
from infinicore.tensor import Tensor


def meshgrid(*tensors: Tensor, indexing: str = "ij") -> tuple[Tensor, ...]:
    if infinicore.use_ntops and tensors[0].device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.meshgrid(*tensors, indexing=indexing)

    return tuple(
        Tensor(t)
        for t in _infinicore.meshgrid(
            [t._underlying for t in tensors], indexing
        )
    )
