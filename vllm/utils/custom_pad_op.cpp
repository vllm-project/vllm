#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <Python.h>
#include <stdexcept>

namespace py = pybind11;

/**
 * Zero-allocation in-place padding operation using raw Python C-API.
 * Writes Python list data directly into a pre-allocated tensor's memory.
 *
 * Uses PyList_GET_ITEM and direct type checks to avoid pybind11 reference
 * counting overhead in the inner loop.
 *
 * @param py_x A Python list of lists containing scalar values
 * @param padded_tensor A pre-allocated contiguous CPU tensor of shape [num_rows, max_len]
 * @param max_len The maximum length (width) of each row
 */
void make_pad_inplace(const py::list& py_x, torch::Tensor& padded_tensor, int64_t max_len) {
    // Validate tensor is contiguous and on CPU
    TORCH_CHECK(padded_tensor.is_contiguous(), "padded_tensor must be contiguous");
    TORCH_CHECK(padded_tensor.device().is_cpu(), "padded_tensor must be on CPU");
    TORCH_CHECK(padded_tensor.dim() == 2, "padded_tensor must be 2D");

    const int64_t num_rows = static_cast<int64_t>(py_x.size());
    TORCH_CHECK(padded_tensor.size(0) == num_rows,
                "padded_tensor rows mismatch: expected ", num_rows,
                ", got ", padded_tensor.size(0));
    TORCH_CHECK(padded_tensor.size(1) == max_len,
                "padded_tensor cols mismatch: expected ", max_len,
                ", got ", padded_tensor.size(1));

    // Get raw PyObject pointer to outer list (borrowed reference)
    PyObject* outer_list = py_x.ptr();

    // Dispatch based on scalar type to handle various dtypes
    // AT_DISPATCH_ALL_TYPES_AND3 covers: uint8, int8, int16, int32, int64, Half, BFloat16, Bool
    AT_DISPATCH_ALL_TYPES_AND3(
        at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool,
        padded_tensor.scalar_type(),
        "make_pad_inplace",
        [&]() {
            scalar_t* data_ptr = padded_tensor.data_ptr<scalar_t>();

            for (int64_t i = 0; i < num_rows; ++i) {
                // Use raw C-API: PyList_GET_ITEM returns borrowed reference (no refcount ops)
                PyObject* row_obj = PyList_GET_ITEM(outer_list, i);

                // Validate row is a list
                if (!PyList_Check(row_obj)) {
                    TORCH_CHECK(false, "Row ", i, " is not a Python list");
                }

                const Py_ssize_t row_len = PyList_GET_SIZE(row_obj);
                TORCH_CHECK(static_cast<int64_t>(row_len) <= max_len,
                            "Row ", i, " length ", row_len, " exceeds max_len ", max_len);

                // Write directly to the tensor's memory at offset [i * max_len + j]
                scalar_t* row_ptr = data_ptr + i * max_len;

                for (Py_ssize_t j = 0; j < row_len; ++j) {
                    // Borrowed reference - no refcount overhead
                    PyObject* val_obj = PyList_GET_ITEM(row_obj, j);

                    // Direct type checks with raw C-API (no pybind11 overhead)
                    if (PyBool_Check(val_obj)) {
                        // Python bool is subclass of int, check first
                        row_ptr[j] = static_cast<scalar_t>(val_obj == Py_True);
                    } else if (PyLong_Check(val_obj)) {
                        // Integer type
                        long long val = PyLong_AsLongLong(val_obj);
                        if (val == -1 && PyErr_Occurred()) {
                            PyErr_Clear();
                            TORCH_CHECK(false, "Row ", i, " element ", j,
                                        " exceeds long long range");
                        }
                        row_ptr[j] = static_cast<scalar_t>(val);
                    } else if (PyFloat_Check(val_obj)) {
                        // Float type
                        double val = PyFloat_AS_DOUBLE(val_obj);
                        row_ptr[j] = static_cast<scalar_t>(val);
                    } else {
                        // Unsupported type - fail fast
                        TORCH_CHECK(false, "Row ", i, " element ", j,
                                    " has unsupported type: ",
                                    Py_TYPE(val_obj)->tp_name);
                    }
                }
                // Remaining elements already filled by torch::full, no need to pad
            }
        }
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Zero-allocation padding operation for vLLM using raw Python C-API";
    m.def("make_pad_inplace", &make_pad_inplace,
          "In-place fill a pre-allocated tensor with padded list data",
          py::arg("py_x"), py::arg("padded_tensor"), py::arg("max_len"));
}