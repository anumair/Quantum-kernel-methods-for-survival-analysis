Median test C-index over 20 splits (kernels: hybrid form; baselines: Cox-LASSO, RSF).

|                           |     vlc |   gbsg2 |   synth-linear |   synth-interaction |   synth-smooth-periodic |   synth-periodic |
|:--------------------------|--------:|--------:|---------------:|--------------------:|------------------------:|-----------------:|
| ('P-Q-2L', 'hybrid')      | nan     | nan     |          0.789 |               0.758 |                   0.744 |            0.586 |
| ('Q-1L', 'hybrid')        | nan     | nan     |          0.793 |               0.77  |                   0.739 |            0.555 |
| ('Q-1L-wrap', 'hybrid')   |   0.693 |   0.645 |        nan     |               0.73  |                   0.739 |            0.716 |
| ('Q-2L', 'hybrid')        | nan     | nan     |          0.795 |               0.769 |                   0.734 |            0.55  |
| ('Q-2L-wrap', 'hybrid')   |   0.61  |   0.636 |        nan     |               0.644 |                   0.715 |            0.659 |
| ('Q-3L', 'hybrid')        | nan     | nan     |          0.793 |               0.765 |                   0.737 |            0.556 |
| ('Q-Z', 'hybrid')         | nan     | nan     |          0.793 |               0.767 |                   0.735 |            0.548 |
| ('Q-ZZ', 'hybrid')        | nan     | nan     |          0.793 |               0.768 |                   0.737 |            0.543 |
| ('Q-ZZ-wrap', 'hybrid')   |   0.606 |   0.636 |        nan     |               0.664 |                   0.717 |            0.52  |
| ('clinical', 'hybrid')    | nan     | nan     |          0.788 |               0.475 |                   0.74  |            0.686 |
| ('cox-lasso', 'baseline') |   0.718 |   0.673 |          0.796 |               0.5   |                   0.745 |            0.569 |
| ('linear', 'hybrid')      | nan     | nan     |          0.795 |               0.497 |                   0.744 |            0.552 |
| ('rbf', 'hybrid')         | nan     | nan     |          0.792 |               0.771 |                   0.737 |            0.547 |
| ('rbf-narrow', 'hybrid')  | nan     | nan     |        nan     |               0.743 |                 nan     |            0.534 |
| ('rsf', 'baseline')       |   0.693 |   0.689 |          0.773 |               0.67  |                   0.738 |            0.67  |