from infinicore.lib import _infinicore


class dtype:
    def __init__(self, data_type):
        """An internal method. Please do not use this directly."""
        self._underlying = data_type

    def __repr__(self):
        repr_map = {
            _infinicore.DataType.BYTE: "uint8",
            _infinicore.DataType.BOOL: "bool",
            _infinicore.DataType.I8: "int8",
            _infinicore.DataType.I16: "int16",
            _infinicore.DataType.I32: "int32",
            _infinicore.DataType.I64: "int64",
            _infinicore.DataType.U8: "uint8",
            _infinicore.DataType.U16: "uint16",
            _infinicore.DataType.U32: "uint32",
            _infinicore.DataType.U64: "uint64",
            _infinicore.DataType.F8: "float8",
            _infinicore.DataType.F16: "float16",
            _infinicore.DataType.F32: "float32",
            _infinicore.DataType.F64: "float64",
            _infinicore.DataType.C16: "complex16",
            _infinicore.DataType.C32: "complex32",
            _infinicore.DataType.C64: "complex64",
            _infinicore.DataType.C128: "complex128",
            _infinicore.DataType.BF16: "bfloat16",
        }
        return f"infinicore.{repr_map[self._underlying]}"

    def __eq__(self, other):
        """
        Compare two dtype objects for equality.

        Args:
            other: The object to compare with

        Returns:
            bool: True if both objects are dtype instances with the same underlying data type
        """
        if not isinstance(other, dtype):
            return False
        return self._underlying == other._underlying

    def __hash__(self):
        """
        Return a hash value for the dtype object.

        Returns:
            int: Hash value based on the underlying data type
        """
        return hash(self._underlying)


float32 = dtype(_infinicore.DataType.F32)
float = float32
float64 = dtype(_infinicore.DataType.F64)
double = float64
complex32 = dtype(_infinicore.DataType.C32)
chalf = complex32
complex64 = dtype(_infinicore.DataType.C64)
cfloat = complex64
complex128 = dtype(_infinicore.DataType.C128)
cdouble = complex128
float16 = dtype(_infinicore.DataType.F16)
half = float16
bfloat16 = dtype(_infinicore.DataType.BF16)
uint8 = dtype(_infinicore.DataType.U8)
int8 = dtype(_infinicore.DataType.I8)
int16 = dtype(_infinicore.DataType.I16)
short = int16
int32 = dtype(_infinicore.DataType.I32)
int = int32
int64 = dtype(_infinicore.DataType.I64)
long = int64
bool = dtype(_infinicore.DataType.BOOL)


def promote_types(type1, type2):
    if type1 == type2:
        return type1

    if type1 in (complex128, complex64, complex32) or type2 in (
        complex128,
        complex64,
        complex32,
    ):
        raise TypeError("promote_types does not support complex dtypes yet")

    if type1 == float64 or type2 == float64:
        return float64
    if type1 == float32 or type2 == float32:
        return float32
    if (type1 == float16 and type2 == bfloat16) or (
        type1 == bfloat16 and type2 == float16
    ):
        return float32
    if type1 == bfloat16 or type2 == bfloat16:
        return bfloat16
    if type1 == float16 or type2 == float16:
        return float16

    if type1 == bool:
        return type2
    if type2 == bool:
        return type1

    signed_integer_order = [int8, int16, int32, int64]
    if type1 == uint8 and type2 == uint8:
        return uint8
    if type1 == uint8 and type2 in signed_integer_order:
        return int16 if type2 == int8 else type2
    if type2 == uint8 and type1 in signed_integer_order:
        return int16 if type1 == int8 else type1
    if type1 in signed_integer_order and type2 in signed_integer_order:
        return signed_integer_order[
            max(
                signed_integer_order.index(type1),
                signed_integer_order.index(type2),
            )
        ]

    raise TypeError(f"Unsupported dtype promotion: {type1}, {type2}")
