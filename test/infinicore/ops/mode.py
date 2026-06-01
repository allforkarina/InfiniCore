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
    ((4, 10), -1),
    ((8, 16), 0),
    ((2, 3, 4), -1),
    ((5, 7), 1),
    ((3, 8, 6), 1),
    ((16, 32), -1),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, dim in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-4})
            in_spec = TensorSpec.from_tensor(shape, None, dtype)

            kwargs = {"dim": dim}
            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    tolerance=tol,
                    output_count=2,
                    description=f"mode({shape}, dim={dim}) - OUT_OF_PLACE",
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
