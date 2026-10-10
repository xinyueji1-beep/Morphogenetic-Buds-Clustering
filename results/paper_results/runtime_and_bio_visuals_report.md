# Runtime and Biological Visualization Report

## Runtime Scaling

|   n_samples |   Agglomerative |   DBSCAN |    GMM |   KMeans |    MBC |   Spectral |
|------------:|----------------:|---------:|-------:|---------:|-------:|-----------:|
|         200 |          0.001  |   0.0025 | 0.0047 |   0.054  | 0.2072 |     0.0299 |
|         500 |          0.004  |   0.004  | 0.0054 |   0.067  | 0.3091 |     0.0666 |
|        1000 |          0.0156 |   0.0075 | 0.0062 |   0.0836 | 0.5472 |     0.1852 |
|        2000 |          0.0731 |   0.0157 | 0.0054 |   0.0299 | 1.2631 |     0.5237 |

## ARI During Runtime Scaling

|   n_samples |   Agglomerative |   DBSCAN |    GMM |   KMeans |    MBC |   Spectral |
|------------:|----------------:|---------:|-------:|---------:|-------:|-----------:|
|         200 |          0.5928 |   0.6254 | 0.4876 |   0.483  | 0.9933 |     0.8662 |
|         500 |          0.6876 |   0.5169 | 0.4871 |   0.4725 | 1      |     0.9153 |
|        1000 |          0.5916 |   0.592  | 0.499  |   0.484  | 1      |     1      |
|        2000 |          0.643  |   0.7765 | 0.4972 |   0.4818 | 1      |     0.9855 |

## Biological PCA Visualizations

| dataset         |    ari |   n_samples |   n_features_after_preprocess |
|:----------------|-------:|------------:|------------------------------:|
| Leukemia_3class | 0.2019 |          72 |                            12 |
| Lymphoma_3class | 0.4109 |          66 |                            12 |

## Figures

- `runtime_scaling.png`
- `bio_visual_Leukemia_3class.png`
- `bio_visual_Lymphoma_3class.png`