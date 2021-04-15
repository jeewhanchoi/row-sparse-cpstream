#include "splatt_debug.h"

void print_matrix_(
    matrix_t * A
) {
    // Only works for 2D matrices for now
    for (idx_t i = 0; i < A->I; i++) {
        fprintf(stderr, "[");
        for (idx_t j = 0; j < A->J; j++) {
            fprintf(stderr, "%e", A->vals[A->J * i+ j]);
            if (j+1 < A->J) fprintf(stderr, ",");
	    else {
	        fprintf(stderr, "],");
	    }
        }
        fprintf(stderr, "\n");
    }
}

/* Prints matrix with label */
void print_matrix(
    char * label,
    matrix_t * mat)
{
  printf("\n\n %s \n\n\n", label);
  print_matrix_(mat);
  printf("\n\n");
}


bool is_matrix_equal(matrix_t *A, matrix_t *B) {
  bool is_eq = true;

  if (A->I != B->I) {
    printf("columns do not match: A: %d, B: %d\n", A->I, B->I);
    is_eq = false;
    return is_eq;
  }
  if (A->J != B->J) {
    printf("rows do not match: A: %d, B: %d\n", A->J, B->J);
    is_eq = false;
    return is_eq;
  }
  for (int i = 0; i < A->I * A->J; i++) {
    // if (A->vals[i] != B->vals[i]) {
    if (abs(A->vals[i] - B->vals[i]) > 1e-9) {
      is_eq = false;
    }
  }

  if (is_eq) {
    return is_eq;
  } else {
    printf("\nPrinting A\n\n");
    print_matrix_(A);
    printf("\nPrinting B\n\n");
    print_matrix_(B);
    printf("\n\n");
    return is_eq;
  }
}

matrix_t* copy_matrix(matrix_t * m) {
  matrix_t* mat = (matrix_t *) splatt_malloc(sizeof(matrix_t));

  mat->I = m->I;
  mat->J = m->J;

  mat->vals = (val_t *) splatt_malloc(m->I * m->J * sizeof(val_t));
  memcpy(mat->vals, m->vals, m->I * m->J * sizeof(val_t));
  mat->rowmajor = 1;

  return mat;
}
