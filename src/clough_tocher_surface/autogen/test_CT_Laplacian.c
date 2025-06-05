
#include "Clough_Tocher_Laplacian.c"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double test_cp[3][10] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

double evaluate_E(double u[3], double v[3], double BM[3][3][10][10],
                  double test_cp[3][10]) {
  double AT[3][10][10];
  double T[3][2][2];
  double detT[3];
  int i, j, n;
  double E[3] = {0.0, 0.0, 0.0};

  compute_AT_from_uv(u, v, 0, BM, AT[0]);
  compute_AT_from_uv(u, v, 1, BM, AT[1]);
  compute_AT_from_uv(u, v, 2, BM, AT[2]);
  compute_UV_2_bary_subtri(u, v, T);
  for (n = 0; n < 3; n++)
    detT[n] = T[n][0][0] * T[n][1][1] - T[n][0][1] * T[n][1][0];

  for (n = 0; n < 3; n++) {
    for (i = 0; i < 10; i++) {
      for (j = 0; j < 10; j++) {
        E[n] += test_cp[n][i] * test_cp[n][j] * AT[n][i][j];
      }
    }
    E[n] /= detT[n];
  }
  return E[0] + E[1] + E[2];
}

int main() {
  double BM[3][3][10][10];
  compute_BM(BM);

  double u1[3] = {0.0, 1.0, 0.0};
  double v1[3] = {0.0, 0.0, 1.0};
  double test_cp1[3][10] = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  printf("E = %g, value should be 0\n", evaluate_E(u1, v1, BM, test_cp1));

  double u2[3] = {-1.0, 1.0, 0.0};
  double v2[3] = {0.0, 0.0, 1.0};
  double test_cp2[3][10] = {
      {-3., -2., -1.296296296, 0., -1., -0.6666666667, 1.296296296,
       -0.3333333333, 3.333333333, 7.},
      {7., 3.333333333, 1.296296296, 0., 3.333333333, 1.222222222, 0., 1., 0.,
       0.},
      {0., 0., 0., 0., -1., -1.222222222, -1.296296296, -2., -2., -3.}};

  printf("E = %g, value from Maple 24\n", evaluate_E(u2, v2, BM, test_cp2));
  double u3[3] = {-1, 2., 1 / 2.};
  double v3[3] = {1 / 2., -1 / 2., 3};

  double test_cp3[3][10] = {
      {-2.750000000, -1.208333333, -0.5833333333, 2.125000000, -0.3333333333,
       -0.4583333333, 5.500000000, 0.4166666667, 12.62500000, 23.50000000},
      {23.50000000, 12.62500000, 5.500000000, 2.125000000, 13.95833333,
       4.333333333, 1.458333333, 0.5000000000, 0.1250000000, -1.875000000},
      {-1.875000000, 0.1250000000, 1.458333333, 2.125000000, 1.750000000,
       -0.08333333333, -0.5833333333, -0.5416666667, -1.208333333,
       -2.750000000}};

  printf("E = %g, value from Maple 348\n", evaluate_E(u3, v3, BM, test_cp3));

  srand(57);
  int i;
  u3[0] = 2.0 * rand() / (double)RAND_MAX - 1.0;
  u3[1] = 2.0 * rand() / (double)RAND_MAX - 1.0;
  u3[2] = 2.0 * rand() / (double)RAND_MAX - 1.0;
  v3[0] = 2.0 * rand() / (double)RAND_MAX - 1.0;
  v3[1] = 2.0 * rand() / (double)RAND_MAX - 1.0;
  v3[2] = 2.0 * rand() / (double)RAND_MAX - 1.0;

  clock_t start = clock();
  for (i = 0; i < 1000000; i++)
    evaluate_E(u3, v3, BM, test_cp3);
  clock_t end = clock();

  printf("Elapsed time: %.6f seconds\n",
         (double)(end - start) / CLOCKS_PER_SEC);
}
