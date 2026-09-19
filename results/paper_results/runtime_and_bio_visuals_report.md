# Runtime and Biological Visualization Report

## Runtime Scaling

|   n_samples |   Agglomerative |   DBSCAN |    GMM |   KMeans |    MBC |   Spectral |
|------------:|----------------:|---------:|-------:|---------:|-------:|-----------:|
|         200 |          0.0016 |   0.0037 | 0.0062 |   0.0772 | 0.3064 |     0.0394 |
|         500 |          0.0061 |   0.0066 | 0.0082 |   0.0973 | 0.4384 |     0.1019 |
|        1000 |          0.0198 |   0.009  | 0.0091 |   0.1209 | 0.7693 |     0.2415 |
|        2000 |          0.101  |   0.0182 | 0.0084 |   0.0507 | 1.7378 |     0.713  |

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