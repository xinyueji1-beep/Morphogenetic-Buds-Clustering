# Runtime and Biological Visualization Report

## Runtime Scaling

|   n_samples |   Agglomerative |   DBSCAN |    GMM |   KMeans |    MBC |   Spectral |
|------------:|----------------:|---------:|-------:|---------:|-------:|-----------:|
|         200 |          0.0013 |   0.0025 | 0.0055 |   0.067  | 0.2415 |     0.0354 |
|         500 |          0.0049 |   0.0045 | 0.007  |   0.0822 | 0.3506 |     0.0744 |
|        1000 |          0.018  |   0.0076 | 0.0071 |   0.095  | 0.6075 |     0.2104 |
|        2000 |          0.0854 |   0.0161 | 0.0066 |   0.0328 | 1.3673 |     0.5766 |

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