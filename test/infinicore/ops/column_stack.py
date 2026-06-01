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
    ([(5,), (5,)],),
    ([(8,), (8,), (8,)],),
    ([(10,), (10,)],),
    ([(3,), (3,), (3,), (3,)],),
    ([(16,), (16,)],),
    ([(4,), (4,), (4,)],),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shapes = data[0]

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-4})
            specs = [
                TensorSpec.from_tensor(shape, None, dtype) for shape in shapes
            ]

            test_cases.append(
                TestCase(
                    inputs=[tuple(specs)],
                    kwargs={},
                    tolerance=tol,
                    description=f"column_stack({len(shapes)} tensors) - OUT_OF_PLACE",
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
