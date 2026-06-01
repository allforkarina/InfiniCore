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
    ([(4,), (6,)], "ij"),
    ([(3,), (5,), (7,)], "ij"),
    ([(4,), (6,)], "xy"),
    ([(8,), (8,)], "ij"),
    ([(2,), (3,), (4,)], "ij"),
    ([(5,), (10,)], "xy"),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shapes, indexing in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-4})
            specs = [
                TensorSpec.from_tensor(shape, None, dtype) for shape in shapes
            ]

            kwargs = {"indexing": indexing}
            test_cases.append(
                TestCase(
                    inputs=[tuple(specs)],
                    kwargs=kwargs,
                    tolerance=tol,
                    output_count=len(shapes),
                    description=f"meshgrid({len(shapes)} tensors, indexing='{indexing}')",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Meshgrid operator test."""

    def __init__(self):
        super().__init__("Meshgrid")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        tensors, = args
        indexing = kwargs.get("indexing", "ij")
        return torch.meshgrid(*tensors, indexing=indexing)

    def infinicore_operator(self, *args, **kwargs):
        import infinicore.nn.functional as F
        tensors, = args
        indexing = kwargs.get("indexing", "ij")
        return F.meshgrid(*tensors, indexing=indexing)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
