import infinicore


def fliplr(input, *, out=None):
    if out is not None:
        raise NotImplementedError("fliplr does not support out= yet")

    if not infinicore.use_ntops or input.device.type not in ("cuda", "musa"):
        raise NotImplementedError(
            "fliplr is only available via ntops on CUDA/MUSA devices"
        )

    return infinicore.ntops.torch.fliplr(input)
