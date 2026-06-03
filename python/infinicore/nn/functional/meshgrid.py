import infinicore


def meshgrid(*tensors, indexing="ij"):
    if indexing != "ij":
        raise NotImplementedError("meshgrid only supports indexing='ij'")

    if not infinicore.use_ntops or tensors[0].device.type not in ("cuda", "musa"):
        raise NotImplementedError(
            "meshgrid is only available via ntops on CUDA/MUSA devices"
        )

    return infinicore.ntops.torch.meshgrid(*tensors, indexing=indexing)
