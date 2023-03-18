Tax Anomaly Detection Using Economic Data

This project explores how data science can reveal unusual tax collection patterns in Nigeria. It uses historical records from the Federal Inland Revenue Service (FIRS) and other economic indicators to identify moments when tax activity significantly strayed from expected trends.

The goal is simple: help analysts and policymakers notice periods that might suggest irregular reporting, inefficiency, or economic shocks. These insights can guide better tax policy, improve data transparency, and support accountability.

To achieve this, the project uses unsupervised machine learning techniques that detect anomalies in large datasets without needing prior labels. It combines models such as Isolation Forest, Local Outlier Factor, K-Nearest Neighbors, HBOS, and Elliptic Envelope to capture different patterns of irregularity.

Each model’s predictions are compared, and only consistent outliers across multiple algorithms are accepted as genuine anomalies. This consensus approach reduces false alarms and strengthens the reliability of findings.

The result is a clear view of which time periods showed suspicious or unexpected behavior in tax data. Some of these align with known national events, including election seasons and policy changes, which adds context to the patterns observed.