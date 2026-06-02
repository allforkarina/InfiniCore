import infinicore


def mode(input, dim=-1, keepdim=False, *, out=None):
    if out is not None:
        raise NotImplementedError("mode does not support out= yet")

    if keepdim:
        raise NotImplementedError("mode does not support keepdim yet")

    if not infinicore.use_ntops or input.device.type not in ("cuda", "musa"):
        raise NotImplementedError(
            "mode is only available via ntops on CUDA/MUSA devices"
        )

    return infinicore.ntops.torch.mode(input, dim)
