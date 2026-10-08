# yyc200 高光谱数据建模分析

## 数据概况

- 文件格式：ENVI Standard，BSQ 排列，16-bit signed integer，little-endian。
- 尺寸：200 行 × 200 列 × 64 波段，共 40,000 个像元。
- 波长范围：462.1 nm 到 10250.0 nm。
- 数值范围：0 到 4095，整体为 12-bit 风格数字量级，最大值 4095 可视作饱和值。

## 质量检查

- 零值比例超过 20% 的波段：48, 54, 55, 56, 57, 58, 59, 60, 63, 64。
- 本次 PCA/聚类使用 54 个信息较完整的波段：1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 61, 62。
- 建模前对每个波段做了 1%-99% 分位数截断，再进行标准化，以降低饱和值和极端值影响。
- 可见光近似 RGB 使用波段 R=19 (681.4 nm), G=9 (559.6 nm), B=3 (486.2 nm)。
- 饱和像元比例超过 0.5% 的波段：B17=0.56%, B18=1.19%, B20=0.62%。

## PCA 结果

- 前 5 个主成分解释方差：PC1=81.75%, PC2=11.54%, PC3=4.28%, PC4=0.83%, PC5=0.40%。
- 前 3 个主成分累计解释：97.57%。
- 前 5 个主成分累计解释：98.80%。

## 无监督聚类

- 评估了 k=3 到 k=10 的 KMeans；样本轮廓系数最高的 k 为 4 (0.401)。
- 最终聚类图采用 k=6 (0.390)，原因是它与最高轮廓系数差距较小，同时能表达更细的城市地物光谱分区。
- 这些类别是无监督光谱分区，不等同于已有地物标签。

| 类别 | 像元数 | 面积占比 | 平均强度 | 峰值波段 | 峰值波长 |
|---|---:|---:|---:|---:|---:|
| C1 | 11,970 | 29.93% | 666.6 | 18 | 669.6 nm |
| C2 | 3,618 | 9.04% | 1544.9 | 18 | 669.6 nm |
| C3 | 8,706 | 21.77% | 356.3 | 18 | 669.6 nm |
| C4 | 9,521 | 23.80% | 1007.7 | 18 | 669.6 nm |
| C5 | 3,120 | 7.80% | 1164.3 | 28 | 783.8 nm |
| C6 | 3,065 | 7.66% | 921.6 | 24 | 739.2 nm |

## NDVI-like 指数

- 使用 Red=B19 (681.4 nm)，NIR=B30 (805.6 nm)。
- 指数范围：-1.000 到 0.503，均值 -0.139。
- 由于原始数据是数字量级而非明确反射率校正结果，这里只能作为植被/高近红外响应的探索性指标。

## 输出文件

- `results/hyperspectral_exploration/band_stats.csv`：逐波段统计。
- `results/hyperspectral_exploration/k_selection_metrics.csv`：聚类 k 值评估。
- `results/hyperspectral_exploration/cluster_summary.csv`：聚类面积与典型强度。
- `figures/hyperspectral_exploration/quicklook_rgb.png`：近似真彩色快视图。
- `figures/hyperspectral_exploration/pca_rgb.png`：PC1-PC3 合成图。
- `figures/hyperspectral_exploration/pca_variance.png`：PCA 解释方差。
- `figures/hyperspectral_exploration/kmeans_cluster_map.png`：KMeans 聚类分区图。
- `figures/hyperspectral_exploration/cluster_spectra.png`：各聚类平均光谱。
- `figures/hyperspectral_exploration/ndvi_like.png`：NDVI-like 指数图。

## 下一步建议

如果有地面样本、分类标签或 ROI，可在当前脚本基础上改成监督分类，例如 Random Forest、SVM 或 XGBoost，并用混淆矩阵、OA、Kappa、F1 等指标评价。
