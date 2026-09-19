# Paper Experiment Report for Morphogenetic Buds Clustering

## Experimental Design

- Seeds: [3, 7, 11, 19, 23]
- Datasets (15): anisotropic_blobs, breast_cancer, circles, cmc, digits_0_4, ecoli, glass, iris, noisy_circles, noisy_moons, two_moons, varied_density, vehicle, wdbc, wine.
- Baselines: KMeans, GMM, Agglomerative, Spectral, DBSCAN.
- Metrics: ARI, NMI, silhouette, Davies-Bouldin, Calinski-Harabasz, runtime.
- Ground-truth labels are used only for external evaluation, never during clustering.

## Main Mean ARI Ranking

| algorithm         |   mean |    std |
|:------------------|-------:|-------:|
| Spectral          | 0.6323 | 0.3206 |
| MBC               | 0.5969 | 0.3668 |
| Agglomerative     | 0.4747 | 0.3228 |
| KMeans            | 0.4527 | 0.32   |
| CompactRefineOnly | 0.4491 | 0.317  |
| GMM               | 0.439  | 0.375  |
| DBSCAN            | 0.421  | 0.2453 |

## Average Rank by ARI

| algorithm         |   mean |    std |
|:------------------|-------:|-------:|
| Spectral          | 3      | 2.1297 |
| MBC               | 3.5667 | 2.0777 |
| Agglomerative     | 3.6    | 1.7238 |
| KMeans            | 4.1667 | 1.3973 |
| GMM               | 4.2    | 2.4842 |
| DBSCAN            | 4.6    | 2.1974 |
| CompactRefineOnly | 4.8667 | 1.3819 |

## Mean ARI by Dataset Family

| family              |   Agglomerative |   CompactRefineOnly |   DBSCAN |    GMM |   KMeans |    MBC |   Spectral |
|:--------------------|----------------:|--------------------:|---------:|-------:|---------:|-------:|-----------:|
| nonconvex           |          0.296  |              0.2367 |   0.5897 | 0.2444 |   0.2369 | 0.92   |     0.879  |
| real_image_features |          0.8325 |              0.7489 |   0.6499 | 0.7018 |   0.7795 | 0.7335 |     0.8838 |
| real_low_dim        |          0.6153 |              0.6201 |   0.4833 | 0.9039 |   0.6201 | 0.6161 |     0.5821 |
| real_tabular        |          0.4185 |              0.4253 |   0.2186 | 0.2979 |   0.4281 | 0.4313 |     0.444  |
| synthetic_density   |          0.6005 |              0.4879 |   0.5144 | 0.955  |   0.4905 | 0.0151 |     0.512  |
| synthetic_gaussian  |          0.9587 |              0.956  |   0.7788 | 0.9618 |   0.956  | 0.8895 |     0.883  |

## MBC Strength Profile

| family              |    MBC |
|:--------------------|-------:|
| nonconvex           | 0.92   |
| synthetic_gaussian  | 0.8895 |
| real_image_features | 0.7335 |
| real_low_dim        | 0.6161 |
| real_tabular        | 0.4313 |
| synthetic_density   | 0.0151 |

## Wilcoxon Tests Against MBC

| comparison               |   mbc_mean |   other_mean |   mean_difference |   wilcoxon_statistic |   p_value |
|:-------------------------|-----------:|-------------:|------------------:|---------------------:|----------:|
| MBC vs KMeans            |     0.5969 |       0.4527 |            0.1442 |                   25 |    0.4769 |
| MBC vs GMM               |     0.5969 |       0.439  |            0.1579 |                   42 |    0.3065 |
| MBC vs Agglomerative     |     0.5969 |       0.4747 |            0.1222 |                   48 |    0.4954 |
| MBC vs Spectral          |     0.5969 |       0.6323 |           -0.0354 |                   39 |    0.6495 |
| MBC vs DBSCAN            |     0.5969 |       0.421  |            0.1759 |                   23 |    0.0356 |
| MBC vs CompactRefineOnly |     0.5969 |       0.4491 |            0.1478 |                   35 |    0.2718 |

## Friedman Test and Nemenyi Critical Difference

- Datasets (N=15), Algorithms (k=7)
- Friedman chi-squared = 8.2892
- Iman-Davenport F = 1.4202, p-value = 0.2177
- Nemenyi critical difference (CD) = 2.3262 (average-rank units)
- Algorithms whose average rank differs from the best by more than CD are flagged significant_vs_best above.

| algorithm         |   average_rank |   diff_from_best | significant_vs_best   |
|:------------------|---------------:|-----------------:|:----------------------|
| Spectral          |         3      |           0      | False                 |
| MBC               |         3.5667 |           0.5667 | False                 |
| Agglomerative     |         3.6    |           0.6    | False                 |
| KMeans            |         4.1667 |           1.1667 | False                 |
| GMM               |         4.2    |           1.2    | False                 |
| DBSCAN            |         4.6    |           1.6    | False                 |
| CompactRefineOnly |         4.8667 |           1.8667 | False                 |

## Ablation Mean ARI

| algorithm        |   mean |    std |
|:-----------------|-------:|-------:|
| MBC_no_capillary | 0.5975 | 0.3652 |
| MBC              | 0.5969 | 0.3668 |
| MBC_no_polarity  | 0.5255 | 0.3847 |
| MBC_no_compact   | 0.4132 | 0.4141 |
| MBC_micro_only   | 0.3321 | 0.2548 |

## Interpretation

MBC is strongest on nonconvex continuous structures, where morphogenetic continuity is useful.
It is weaker on some high-dimensional tabular datasets because the current organ-fusion rule can over-segment compact classes.
This supports positioning MBC as a morphology-aware clustering method rather than a universal replacement for all clustering algorithms.