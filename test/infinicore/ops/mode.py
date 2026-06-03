import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TensorInitializer,
    TestCase,
    GenericTestRunner,
)

_HAND_CRAFTED = [
    (
        "2d_dim1",
        torch.tensor([[1, 2, 2, 1], [3, 3, 2, 2]], dtype=torch.float32),
        1,
    ),
    (
        "2d_dim-1",
        torch.tensor([[2, 1, 1, 2], [2, 2, 1, 1]], dtype=torch.float32),
        -1,
    ),
    (
        "3d_dim2",
        torch.tensor(
            [[[1, 1, 2], [4, 5, 5]], [[1, 2, 2], [4, 4, 5]]], dtype=torch.float32
        ),
        2,
    ),
    (
        "3d_dim1",
        torch.tensor(
            [[[1, 3], [1, 4], [2, 4]], [[5, 5], [6, 5], [6, 7]]], dtype=torch.float32
        ),
        1,
    ),
    (
        "3d_dim0",
        torch.tensor(
            [[[1, 8], [3, 4]], [[1, 9], [3, 4]], [[2, 9], [5, 4]]], dtype=torch.float32
        ),
        0,
    ),
    (
        "large_2d_dim-1",
        torch.tensor(
            [[0] * 12 + [1] * 12 + [2] * 8],
            dtype=torch.float32,
        ),
        -1,
    ),
]

_RANDOM_CASES = [
    ((4, 10), -1),
    ((8, 16), 0),
    ((2, 3, 4), -1),
    ((5, 7), 1),
    ((3, 8, 6), 1),
]

_STRIDE_CASES = [
    ((4, 10), -1, (30, 1)),
    ((8, 16), 0, (128, 8)),
    ((3, 8, 6), 1, (192, 24, 1)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []

    for name, data, dim in _HAND_CRAFTED:
        shape = tuple(data.shape)
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-4})
            in_spec = TensorSpec.from_tensor(
                shape,
                None,
                dtype,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=data,
            )
            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={"dim": dim},
                    tolerance=tol,
                    output_count=2,
                    description=f"mode({name}) - OUT_OF_PLACE",
                )
            )

    for shape, dim in _RANDOM_CASES:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-4})
            in_spec = TensorSpec.from_tensor(shape, None, dtype)
            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={"dim": dim},
                    tolerance=tol,
                    output_count=2,
                    description=f"mode({shape}, dim={dim}) - OUT_OF_PLACE",
                )
            )

    for shape, dim, strides in _STRIDE_CASES:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-4})
            in_spec = TensorSpec.from_tensor(shape, strides, dtype)
            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={"dim": dim},
                    tolerance=tol,
                    output_count=2,
                    description=f"mode({shape}, dim={dim}, strided) - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Mode operator test."""

    def __init__(self):
        super().__init__("Mode")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        result = torch.mode(*args, **kwargs)
        return result.values, result.indices

    def infinicore_operator(self, *args, **kwargs):
        import infinicore.nn.functional as F

        return F.mode(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
