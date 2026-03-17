#define PY_SSIZE_T_CLEAN
#include <Python.h>

namespace {

int mirror_square(int square) {
    return square ^ 56;
}

PyObject* py_move_signature(PyObject*, PyObject* args) {
    int from_square = 0;
    int to_square = 0;
    int promotion = 0;
    if (!PyArg_ParseTuple(args, "ii|i", &from_square, &to_square, &promotion)) {
        return nullptr;
    }

    long signature = static_cast<long>(from_square)
        | (static_cast<long>(to_square) << 6)
        | (static_cast<long>(promotion) << 12);
    return PyLong_FromLong(signature);
}

PyObject* py_move_to_idx(PyObject*, PyObject* args) {
    int from_square = 0;
    int to_square = 0;
    int promotion = 0;
    int is_white_turn = 1;
    if (!PyArg_ParseTuple(args, "iiii", &from_square, &to_square, &promotion, &is_white_turn)) {
        return nullptr;
    }

    if (!is_white_turn) {
        from_square = mirror_square(from_square);
        to_square = mirror_square(to_square);
    }

    const int from_rank = from_square / 8;
    const int from_file = from_square % 8;
    const int to_rank = to_square / 8;
    const int to_file = to_square % 8;
    const int dr = to_rank - from_rank;
    const int df = to_file - from_file;

    if (promotion != 0 && promotion != 5) {
        int piece_offset = 0;
        if (promotion == 2) {
            piece_offset = 0;
        } else if (promotion == 3) {
            piece_offset = 1;
        } else if (promotion == 4) {
            piece_offset = 2;
        } else {
            PyErr_SetString(PyExc_ValueError, "unsupported promotion piece");
            return nullptr;
        }

        const int direction = df + 1;
        const int plane_idx = 64 + piece_offset * 3 + direction;
        return PyLong_FromLong(from_square * 73 + plane_idx);
    }

    static const int knight_moves[8][2] = {
        {2, 1}, {1, 2}, {-1, 2}, {-2, 1},
        {-2, -1}, {-1, -2}, {1, -2}, {2, -1},
    };
    for (int i = 0; i < 8; ++i) {
        if (dr == knight_moves[i][0] && df == knight_moves[i][1]) {
            return PyLong_FromLong(from_square * 73 + 56 + i);
        }
    }

    int dir_idx = -1;
    if (dr > 0 && df == 0) {
        dir_idx = 0;
    } else if (dr < 0 && df == 0) {
        dir_idx = 1;
    } else if (dr == 0 && df > 0) {
        dir_idx = 2;
    } else if (dr == 0 && df < 0) {
        dir_idx = 3;
    } else if (dr > 0 && df > 0) {
        dir_idx = 4;
    } else if (dr > 0 && df < 0) {
        dir_idx = 5;
    } else if (dr < 0 && df > 0) {
        dir_idx = 6;
    } else if (dr < 0 && df < 0) {
        dir_idx = 7;
    }

    if (dir_idx < 0) {
        PyErr_SetString(PyExc_ValueError, "unsupported move direction");
        return nullptr;
    }

    int abs_dr = dr >= 0 ? dr : -dr;
    int abs_df = df >= 0 ? df : -df;
    const int dist = abs_dr > abs_df ? abs_dr : abs_df;
    const int plane_idx = dir_idx * 7 + (dist - 1);
    return PyLong_FromLong(from_square * 73 + plane_idx);
}

PyMethodDef kMethods[] = {
    {"move_signature", py_move_signature, METH_VARARGS, "Pack a move into a compact integer signature."},
    {"move_to_idx", py_move_to_idx, METH_VARARGS, "Convert a move into the AlphaZero policy index."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "_speedups",
    "Optional native helpers for TeenyZero hot paths.",
    -1,
    kMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit__speedups(void) {
    return PyModule_Create(&kModule);
}
