import infinicore
from infinicore.tensor import Tensor


def roll(input: Tensor, shifts: int, dims: int = 0, *, out=None) -> Tensor:
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.roll(input, shifts, dims)

    if out is None:
        return Tensor(_infinicore.roll(input._underlying, shifts, dims))

    _infinicore.roll_(out._underlying, input._underlying, shifts, dims)
    return out
