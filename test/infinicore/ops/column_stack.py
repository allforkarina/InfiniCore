import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
)

_TEST_CASES_DATA = [
    ("scalars", [((), None), ((), None)]),
    ("single_1d", [((5,), None)]),
    ("two_1d", [((8,), None), ((8,), None)]),
    ("three_1d", [((7,), None), ((7,), None), ((7,), None)]),
    ("single_2d", [((4, 3), None)]),
    ("two_2d", [((4, 2), None), ((4, 5), None)]),
    ("2d_and_1d", [((4, 2), None), ((4,), None)]),
    ("two_3d", [((2, 3, 4), None), ((2, 2, 4), None)]),
    ("strided_1d", [((4,), (2,)), ((4,), (3,))]),
    ("strided_2d_and_1d", [((4, 2), (1, 4)), ((4,), (2,))]),
]

_MIXED_DTYPE_CASES_DATA = [
    (
        "mixed_float16_float32_1d",
        [
            ((6,), None, infinicore.float16),
            ((6,), None, infinicore.float32),
        ],
    ),
    (
        "mixed_bfloat16_float32_2d",
        [
            ((4, 2), None, infinicore.bfloat16),
            ((4, 3), None, infinicore.float32),
        ],
    ),
    (
        "mixed_float16_float32_3d",
        [
            ((2, 2, 4), None, infinicore.float16),
            ((2, 1, 4), None, infinicore.float32),
        ],
    ),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def _make_specs(specs_data, dtype):
    return [
        TensorSpec.from_tensor(shape, strides, dtype)
        for shape, strides in specs_data
    ]


def _make_mixed_specs(specs_data):
    return [
        TensorSpec.from_tensor(shape, strides, dtype)
        for shape, strides, dtype in specs_data
    ]


def parse_test_cases():
    test_cases = []
    for name, specs_data in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-4})
            specs = _make_specs(specs_data, dtype)

            test_cases.append(
                TestCase(
                    inputs=[tuple(specs)],
                    kwargs={},
                    tolerance=tol,
                    description=f"column_stack({name}, {len(specs)} tensors) - OUT_OF_PLACE",
                )
            )

    for name, specs_data in _MIXED_DTYPE_CASES_DATA:
        specs = _make_mixed_specs(specs_data)
        test_cases.append(
            TestCase(
                inputs=[tuple(specs)],
                kwargs={},
                tolerance=_TOLERANCE_MAP[infinicore.float32],
                description=f"column_stack({name}, {len(specs)} tensors) - OUT_OF_PLACE",
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    """ColumnStack operator test."""

    def __init__(self):
        super().__init__("ColumnStack")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        tensors, = args
        return torch.column_stack(tuple(tensors))

    def infinicore_operator(self, *args, **kwargs):
        import infinicore.nn.functional as F
        tensors, = args
        return F.column_stack(tuple(tensors))


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
