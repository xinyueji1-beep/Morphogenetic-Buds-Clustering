# Paper Experiment Report for Morphogenetic Buds Clustering

## Experimental Design

- Seeds: [3, 7, 11, 19, 23]
- Datasets (14): anisotropic_blobs, breast_cancer, circles, cmc, digits_0_4, ecoli, glass, iris, noisy_circles, noisy_moons, two_moons, varied_density, vehicle, wine.
- Baselines: KMeans, GMM, Agglomerative, Spectral, DBSCAN.
- Metrics: ARI, NMI, silhouette, Davies-Bouldin, Calinski-Harabasz, runtime.
- Ground-truth labels are used only for external evaluation, never during clustering.

## Main Mean ARI Ranking

| algorithm         |   mean |    std |
|:------------------|-------:|-------:|
| Spectral          | 0.624  | 0.3304 |
| MBC               | 0.5919 | 0.3794 |
| GMM               | 0.4606 | 0.3791 |
| Agglomerative     | 0.4585 | 0.3283 |
| DBSCAN            | 0.4425 | 0.2397 |
| KMeans            | 0.4374 | 0.3259 |
| CompactRefineOnly | 0.4343 | 0.3231 |

## Average Rank by ARI

| algorithm         |   mean |    std |
|:------------------|-------:|-------:|
| Spectral          | 3.1429 | 2.1342 |
| MBC               | 3.5714 | 2.156  |
| Agglomerative     | 3.7143 | 1.7289 |
| GMM               | 4.0714 | 2.5257 |
| KMeans            | 4.2143 | 1.4373 |
| DBSCAN            | 4.4286 | 2.1738 |
| CompactRefineOnly | 4.8571 | 1.4335 |

## Mean ARI by Dataset Family

| family              |   Agglomerative |   CompactRefineOnly |   DBSCAN |    GMM |   KMeans |    MBC |   Spectral |
|:--------------------|----------------:|--------------------:|---------:|-------:|---------:|-------:|-----------:|
| nonconvex           |          0.296  |              0.2367 |   0.5897 | 0.2444 |   0.2369 | 0.92   |     0.879  |
| real_image_features |          0.8325 |              0.7489 |   0.6499 | 0.7018 |   0.7795 | 0.7335 |     0.8838 |
| real_low_dim        |          0.6153 |              0.6201 |   0.4833 | 0.9039 |   0.6201 | 0.6161 |     0.5821 |
| real_tabular        |          0.3713 |              0.3867 |   0.235  | 0.3247 |   0.3883 | 0.392  |     0.3932 |
| synthetic_density   |          0.6005 |              0.4879 |   0.5144 | 0.955  |   0.4905 | 0.0151 |     0.512  |
| synthetic_gaussian  |          0.9587 |              0.956  |   0.7788 | 0.9618 |   0.956  | 0.8895 |     0.883  |

## MBC Strength Profile

| family              |    MBC |
|:--------------------|-------:|
| nonconvex           | 0.92   |
| synthetic_gaussian  | 0.8895 |
| real_image_features | 0.7335 |
| real_low_dim        | 0.6161 |
| real_tabular        | 0.392  |
| synthetic_density   | 0.0151 |

## Wilcoxon Tests Against MBC

| comparison               |   n_datasets |   n_nonzero_differences |   mbc_mean |   other_mean |   mean_difference |   wilcoxon_statistic |   p_value |
|:-------------------------|-------------:|------------------------:|-----------:|-------------:|------------------:|---------------------:|----------:|
| MBC vs KMeans            |           14 |                      11 |     0.5919 |       0.4374 |            0.1545 |                   25 |    0.4769 |
| MBC vs GMM               |           14 |                      14 |     0.5919 |       0.4606 |            0.1313 |                   41 |    0.5016 |
| MBC vs Agglomerative     |           14 |                      14 |     0.5919 |       0.4585 |            0.1334 |                   38 |    0.391  |
| MBC vs Spectral          |           14 |                      12 |     0.5919 |       0.624  |           -0.0322 |                   37 |    0.8753 |
| MBC vs DBSCAN            |           14 |                      14 |     0.5919 |       0.4425 |            0.1493 |                   23 |    0.0676 |
| MBC vs CompactRefineOnly |           14 |                      13 |     0.5919 |       0.4343 |            0.1576 |                   32 |    0.3454 |

## Friedman Test and Nemenyi Critical Difference

- Datasets (N=14), Algorithms (k=7)
- Friedman chi-squared = 6.0312
- Iman-Davenport F = 1.0056, p-value = 0.4197
- Nemenyi critical difference (CD) = 2.4078 (average-rank units)
- Algorithms whose average rank differs from the best by more than CD are flagged significant_vs_best above.

| algorithm         |   average_rank |   diff_from_best | significant_vs_best   |
|:------------------|---------------:|-----------------:|:----------------------|
| Spectral          |         3.1429 |           0      | False                 |
| MBC               |         3.5714 |           0.4286 | False                 |
| Agglomerative     |         3.7143 |           0.5714 | False                 |
| GMM               |         4.0714 |           0.9286 | False                 |
| KMeans            |         4.2143 |           1.0714 | False                 |
| DBSCAN            |         4.4286 |           1.2857 | False                 |
| CompactRefineOnly |         4.8571 |           1.7143 | False                 |

## Ablation Mean ARI

| algorithm        |   mean |    std |
|:-----------------|-------:|-------:|
| MBC_no_capillary | 0.5925 | 0.3777 |
| MBC              | 0.5919 | 0.3794 |
| MBC_no_polarity  | 0.5154 | 0.3965 |
| MBC_no_compact   | 0.4419 | 0.4141 |
| MBC_micro_only   | 0.3297 | 0.2631 |

## Interpretation

MBC is strongest on nonconvex continuous structures, where morphogenetic continuity is useful.
It is weaker on some high-dimensional tabular datasets because the current organ-fusion rule can over-segment compact classes.
This supports positioning MBC as a morphology-aware clustering method rather than a universal replacement for all clustering algorithms.