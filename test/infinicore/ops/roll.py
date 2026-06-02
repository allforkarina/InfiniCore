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
    ((10,), 3, 0, None),
    ((10,), -2, 0, None),
    ((4, 8), 2, 1, None),
    ((4, 8), 2, 1, (64, 1)),
    ((2, 3, 4), 1, 0, None),
    ((2, 3, 4), 2, 2, None),
    ((2, 3, 4), -1, -1, None),
    ((16, 64), 8, 0, None),
    ((8, 16, 32), 5, 1, None),
    ((2, 8, 3), 3, -2, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, shifts, dims = data[0], data[1], data[2]
        in_strides = data[3] if len(data) > 3 else None

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-4})
            in_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            kwargs = {"shifts": shifts, "dims": dims}
            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    tolerance=tol,
                    description=f"roll({shape}, shifts={shifts}, dims={dims}) - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Roll operator test."""

    def __init__(self):
        super().__init__("Roll")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.roll(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        import infinicore.nn.functional as F
        return F.roll(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
