import infinicore
from infinicore.tensor import Tensor


def pixel_unshuffle(input: Tensor, downscale_factor: int, *, out=None) -> Tensor:
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.pixel_unshuffle(input, downscale_factor)

    if out is None:
        return Tensor(
            _infinicore.pixel_unshuffle(input._underlying, downscale_factor)
        )

    _infinicore.pixel_unshuffle_(
        out._underlying, input._underlying, downscale_factor
    )
    return out
