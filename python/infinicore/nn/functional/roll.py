import infinicore
from infinicore.tensor import Tensor


def roll(input: Tensor, shifts: int, dims: int = 0, *, out=None) -> Tensor:
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.roll(input, shifts, dims)

    if out is not None:
        raise NotImplementedError("roll with out= is not implemented")

    raise NotImplementedError(
        "roll is currently implemented through ntops on cuda/musa devices only"
    )
