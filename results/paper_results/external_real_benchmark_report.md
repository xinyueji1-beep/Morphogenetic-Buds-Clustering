# External Real-Data Benchmark

## Overall Mean ARI

| algorithm         |   mean |    std |
|:------------------|-------:|-------:|
| GMM               | 0.1534 | 0.1607 |
| Agglomerative     | 0.131  | 0.1077 |
| DBSCAN            | 0.1259 | 0.1718 |
| MBC               | 0.1237 | 0.1232 |
| CompactRefineOnly | 0.1176 | 0.1113 |
| KMeans            | 0.1176 | 0.1096 |
| Spectral          | 0.0894 | 0.138  |

## Biological Dataset: MiceProtein

| algorithm         |   mean |    std |
|:------------------|-------:|-------:|
| Spectral          | 0.2309 | 0      |
| GMM               | 0.2277 | 0.0437 |
| CompactRefineOnly | 0.1368 | 0.0042 |
| KMeans            | 0.1346 | 0.0057 |
| MBC               | 0.1313 | 0.0042 |
| Agglomerative     | 0.1175 | 0      |
| DBSCAN            | 0.0842 | 0      |

## Biological Gene Expression Datasets

| algorithm         |   mean |    std |
|:------------------|-------:|-------:|
| GMM               | 0.2719 | 0.176  |
| Agglomerative     | 0.2558 | 0.1092 |
| MBC               | 0.255  | 0.1317 |
| KMeans            | 0.2319 | 0.1142 |
| CompactRefineOnly | 0.2274 | 0.1225 |
| Spectral          | 0.2183 | 0.1545 |
| DBSCAN            | 0.1496 | 0.1919 |

## Mean ARI by Family

| family                        |   Agglomerative |   CompactRefineOnly |   DBSCAN |     GMM |   KMeans |    MBC |   Spectral |
|:------------------------------|----------------:|--------------------:|---------:|--------:|---------:|-------:|-----------:|
| biological_factor             |          0.0492 |              0.0118 |   0.0662 |  0.0764 |   0.0118 | 0.0121 |     0.0014 |
| biological_gene_expression    |          0.2558 |              0.2274 |   0.1496 |  0.2719 |   0.2319 | 0.255  |     0.2183 |
| biological_protein_expression |          0.1175 |              0.1368 |   0.0842 |  0.2277 |   0.1346 | 0.1313 |     0.2309 |
| real_medical_tabular          |          0.0654 |              0.0913 |   0.0354 |  0.0088 |   0.0825 | 0.0811 |     0.0064 |
| real_mixed_tabular            |          0.0567 |              0.0216 |   0.0365 | -0.0036 |   0.022  | 0.0157 |    -0.0259 |
| real_shape_features           |          0.099  |              0.0754 |   0.0968 |  0.0833 |   0.076  | 0.0754 |     0.1067 |
| real_signal_features          |          0.1286 |              0.1679 |   0.5513 |  0.3527 |   0.1679 | 0.1679 |    -0.0391 |

## ARI by Dataset

| dataset               |   Agglomerative |   CompactRefineOnly |   DBSCAN |     GMM |   KMeans |    MBC |   Spectral |
|:----------------------|----------------:|--------------------:|---------:|--------:|---------:|-------:|-----------:|
| Leukemia_2class       |          0.2685 |              0.2568 |  -0.0349 |  0.1753 |   0.2379 | 0.2568 |     0.0814 |
| Leukemia_3class       |          0.1203 |              0.0938 |  -0.012  |  0.1504 |   0.1295 | 0.1637 |     0.0753 |
| Lung_5class           |          0.2232 |              0.1576 |   0.4092 |  0.2906 |   0.1527 | 0.1492 |     0.2908 |
| Lymphoma_3class       |          0.4112 |              0.4012 |   0.2362 |  0.4715 |   0.4073 | 0.4504 |     0.4257 |
| MiceProtein_8class    |          0.1175 |              0.1368 |   0.0842 |  0.2277 |   0.1346 | 0.1313 |     0.2309 |
| MiceProtein_behavior  |          0.0903 |              0.0307 |   0.1574 |  0.0186 |   0.0307 | 0.0307 |     0.0023 |
| MiceProtein_genotype  |          0.0406 |              0.0042 |   0.0287 |  0.0495 |   0.0042 | 0.0048 |    -0.0022 |
| MiceProtein_treatment |          0.0168 |              0.0005 |   0.0125 |  0.1611 |   0.0005 | 0.0006 |     0.004  |
| blood-transfusion     |          0.0286 |              0.0496 |  -0.0157 | -0.048  |   0.0511 | 0.048  |     0.0008 |
| credit-g              |          0.0567 |              0.0216 |   0.0365 | -0.0036 |   0.022  | 0.0157 |    -0.0259 |
| diabetes              |          0.1022 |              0.1331 |   0.0865 |  0.0656 |   0.1139 | 0.1141 |     0.0119 |
| ionosphere            |          0.1286 |              0.1679 |   0.5513 |  0.3527 |   0.1679 | 0.1679 |    -0.0391 |
| vehicle               |          0.099  |              0.0754 |   0.0968 |  0.0833 |   0.076  | 0.0754 |     0.1067 |

## Files

- Raw results: `external_real_benchmark_raw.csv`
- Summary: `external_real_benchmark_summary.csv`
- Figure: `external_real_benchmark_ari.png`