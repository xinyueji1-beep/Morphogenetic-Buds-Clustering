# Runtime and Biological Visualization Report

## Runtime Scaling

|   n_samples |   Agglomerative |   DBSCAN |    GMM |   KMeans |    MBC |   Spectral |
|------------:|----------------:|---------:|-------:|---------:|-------:|-----------:|
|         200 |          0.0081 |   0.0029 | 0.0055 |   1.6751 | 0.2851 |     0.0391 |
|         500 |          0.0046 |   0.0049 | 0.0067 |   0.0766 | 0.4077 |     0.0809 |
|        1000 |          0.0148 |   0.0086 | 0.0085 |   0.1097 | 0.7097 |     0.2116 |
|        2000 |          0.0717 |   0.0173 | 0.0077 |   0.0548 | 1.5445 |     0.6188 |

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