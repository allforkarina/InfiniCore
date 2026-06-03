import infinicore


def feature_alpha_dropout(input, p=0.5, training=False, *, out=None):
    if out is not None:
        raise NotImplementedError(
            "feature_alpha_dropout does not support out= yet"
        )

    if not infinicore.use_ntops or input.device.type not in ("cuda", "musa"):
        raise NotImplementedError(
            "feature_alpha_dropout is only available via ntops "
            "on CUDA/MUSA devices"
        )

    return infinicore.ntops.torch.feature_alpha_dropout(input, p, training)
