/*
 * _features_module.c — Thin Python C-API wrapper around predted_features.
 *
 * Exposes compute_selected_features() as predted._features_c.compute_features().
 */

#include <Python.h>
#include "predted_features.h"

static PyObject* py_compute_features(PyObject* self, PyObject* args) {
    const char* structure;
    if (!PyArg_ParseTuple(args, "s", &structure))
        return NULL;

    double features[NUM_FEATURES_BASE];
    compute_selected_features(structure, features);

    PyObject* result = PyList_New(NUM_FEATURES_BASE);
    if (!result)
        return NULL;
    for (int i = 0; i < NUM_FEATURES_BASE; i++) {
        PyList_SET_ITEM(result, i, PyFloat_FromDouble(features[i]));
    }
    return result;
}

static PyMethodDef methods[] = {
    {"compute_features", py_compute_features, METH_VARARGS,
     "Compute 36 structural features from a dot-bracket string."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_features_c",
    "C-accelerated feature computation for predted.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__features_c(void) {
    return PyModule_Create(&module);
}
