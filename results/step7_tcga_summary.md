TCGA-BRCA (1063 patients, 148 deaths), median test C-index over 20 splits

| kernel                  | form     |   median |   iqr |
|:------------------------|:---------|---------:|------:|
| Q-Z                     | hybrid   |    0.766 | 0.047 |
| Q-1L                    | hybrid   |    0.766 | 0.039 |
| linear                  | hybrid   |    0.765 | 0.049 |
| Q-2L                    | hybrid   |    0.762 | 0.036 |
| P-Q-2L                  | hybrid   |    0.762 | 0.04  |
| Q-ZZ                    | hybrid   |    0.761 | 0.037 |
| rbf                     | hybrid   |    0.761 | 0.042 |
| clinical                | hybrid   |    0.754 | 0.055 |
| cox-lasso               | baseline |    0.75  | 0.036 |
| cox-lasso-clinical-only | baseline |    0.746 | 0.043 |
| cox-lasso-all-genes     | baseline |    0.724 | 0.036 |
| rsf                     | baseline |    0.715 | 0.057 |

TCGA-BRCA: Spearman(kta, test C) = 0.583 (p = 7.15e-06); event-only 0.602 (p = 2.91e-06); 51 configs

Best config per kernel (3 splits):

| kernel   |   param |   offdiag_mean |   eff_rank_frac |   kta |   align_rbf |   g_vs_rbf |   gap |   test_cindex |
|:---------|--------:|---------------:|----------------:|------:|------------:|-----------:|------:|--------------:|
| P-Q-2L   |   0.1   |          0.399 |           0.007 | 0.043 |       0.938 |      3.979 | 0.068 |         0.749 |
| Q-1L     |   0.464 |          0.738 |           0.003 | 0.049 |       0.917 |      2.686 | 0.04  |         0.76  |
| Q-2L     |   0.316 |          0.728 |           0.003 | 0.046 |       0.92  |      2.615 | 0.051 |         0.759 |
| Q-3L     |   0.316 |          0.624 |           0.003 | 0.042 |       0.923 |      2.106 | 0.055 |         0.759 |
| Q-Z      |   0.464 |          0.68  |           0.003 | 0.045 |       0.931 |      2.575 | 0.052 |         0.756 |
| Q-ZZ     |   0.464 |          0.632 |           0.003 | 0.043 |       0.929 |      2.546 | 0.031 |         0.76  |
| clinical | nan     |          0.8   |           0.002 | 0.04  |       0.885 |     17.454 | 0.029 |         0.751 |
| linear   | nan     |         -0.011 |           0.01  | 0.049 |       0.876 |      4.637 | 0.019 |         0.755 |
| rbf      |   0.25  |          0.765 |           0.002 | 0.048 |       0.942 |      2.782 | 0.038 |         0.758 |
