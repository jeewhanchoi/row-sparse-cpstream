#ifndef SPLATT_DEBUG_H
#define SPLATT_DEBUG_H

#include "base.h"
#include "matrix.h"


void print_matrix(char * label, matrix_t * mat);
void print_matrix_(matrix_t * A);

void dump_matrix_to_file(
    matrix_t * A,
    FILE * fp
);

bool is_matrix_equal(matrix_t *A, matrix_t *B);
matrix_t * copy_matrix(matrix_t *m);

#endif
