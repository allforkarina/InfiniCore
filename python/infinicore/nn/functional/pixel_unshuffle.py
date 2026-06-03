import infinicore


def pixel_unshuffle(input, downscale_factor, *, out=None):
    if out is not None:
        raise NotImplementedError(
            "pixel_unshuffle does not support out= yet"
        )

    if not infinicore.use_ntops or input.device.type not in ("cuda", "musa"):
        raise NotImplementedError(
            "pixel_unshuffle is only available via ntops "
            "on CUDA/MUSA devices"
        )

    return infinicore.ntops.torch.pixel_unshuffle(input, downscale_factor)
