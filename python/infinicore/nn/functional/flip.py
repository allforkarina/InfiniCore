import infinicore


def flip(input, dims, *, out=None):
    if out is not None:
        raise NotImplementedError("flip does not support out= yet")

    if not infinicore.use_ntops or input.device.type not in ("cuda", "musa"):
        raise NotImplementedError(
            "flip is only available via ntops on CUDA/MUSA devices"
        )

    return infinicore.ntops.torch.flip(input, dims)
