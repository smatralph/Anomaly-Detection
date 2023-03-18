## Project Methodology
### 1. Data Collection and Cleaning

The dataset includes tax categories and macroeconomic variables like GDP and population. Missing values were replaced using column means, and all numeric fields were standardized. Unnecessary text or duplicate records were removed to make the data ready for analysis.

### 2. Data Transformation

Most economic indicators are naturally skewed, so a logarithmic transformation was applied to reduce extreme variations and bring values closer to a normal distribution. This step improved model stability.

A new metric called GDP Index was also introduced. It represents GDP per person and helps measure economic productivity across years.

### 3. Dimensionality Reduction

Principal Component Analysis (PCA) was used to capture the main trends in the dataset while reducing redundancy among variables. The top three components were enough to retain most of the important information.

### 4. Model Application

Five different anomaly detection algorithms were used:

Isolation Forest – separates outliers based on random splits in the data.

Local Outlier Factor (LOF) – compares the local density of each point to its neighbors.

K-Nearest Neighbors (KNN) – flags points that lie far from their closest neighbors.

HBOS (Histogram-Based Outlier Score) – detects outliers by measuring histogram frequency deviations.

Elliptic Envelope – fits an ellipse around the main data cluster and marks anything outside it as an outlier.

Each model produced its own list of flagged data points, giving a diverse perspective on potential anomalies.

### 5. Consensus and Evaluation

To improve reliability, the results from all models were combined. A record was considered a confirmed anomaly only if at least three models agreed on it. This rule prevented random noise from being labeled as a problem.

The combined results were visualized to show how anomaly intensity changed over time. Some spikes matched major political or economic events, which gives the findings real-world relevance.