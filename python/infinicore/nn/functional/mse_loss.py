import infinicore
from infinicore.tensor import Tensor


def mse_loss(
    input: Tensor, target: Tensor, reduction: str = "mean", *, out=None
) -> Tensor:
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.mse_loss(input, target, reduction)

    if out is None:
        return Tensor(
            _infinicore.mse_loss(input._underlying, target._underlying, reduction)
        )

    _infinicore.mse_loss_(
        out._underlying, input._underlying, target._underlying, reduction
    )
    return out
