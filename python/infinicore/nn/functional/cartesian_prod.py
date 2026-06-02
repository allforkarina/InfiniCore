import infinicore
from infinicore.tensor import Tensor


def cartesian_prod(*tensors: Tensor, out=None) -> Tensor:
    if (
        infinicore.use_ntops
        and tensors[0].device.type in ("cuda", "musa")
        and out is None
    ):
        return infinicore.ntops.torch.cartesian_prod(*tensors)

    if out is None:
        return Tensor(
            _infinicore.cartesian_prod([t._underlying for t in tensors])
        )

    _infinicore.cartesian_prod_(
        out._underlying, [t._underlying for t in tensors]
    )
    return out
