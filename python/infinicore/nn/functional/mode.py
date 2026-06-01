import infinicore
from infinicore.tensor import Tensor


def mode(input: Tensor, dim: int = -1, *, out=None) -> Tensor:
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.mode(input, dim)

    if out is None:
        return Tensor(_infinicore.mode(input._underlying, dim))

    _infinicore.mode_(out._underlying, input._underlying, dim)
    return out
