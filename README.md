# Pooling Analysis for Document Scanning

## 📌 Problem Statement

A document-scanning application uses pooling after extracting text edges. This project analyzes what information is retained and discarded after pooling, examines the effect of pooling on spatial resolution, and determines whether pooling is appropriate for preserving small characters.

---

## 🎯 Objective

The objectives of this project are:

- To create a sample document containing different text sizes.
- To preprocess the document image.
- To extract text edges using Canny Edge Detection.
- To implement 2×2 Max Pooling.
- To implement 2×2 Average Pooling.
- To analyze information retention and spatial resolution reduction.
- To evaluate the impact of pooling on small and very small characters.
- To compare Max Pooling and Average Pooling.

---

## 🧠 Deep Learning Concept Used

### Pooling

Pooling is a downsampling operation commonly used in Convolutional Neural Networks (CNNs). It reduces the spatial dimensions of feature maps while retaining important information.

### Max Pooling

Max Pooling selects the maximum value from each pooling region.

It helps retain:

- Strong edge activations
- Important boundaries
- Prominent character strokes

However, it may discard:

- Exact spatial locations
- Weak edges
- Fine character details

### Average Pooling

Average Pooling calculates the average value of each pooling region.

It helps retain:

- General feature patterns
- Overall feature information

However, it can weaken:

- Strong edges
- Thin strokes
- Fine character structures

---

## ⚙️ Methodology

The project follows this pipeline:

Document Image
↓
Image Preprocessing
↓
Canny Edge Detection
↓
Original Edge Feature Map
↓
├── 2×2 Max Pooling
│
└── 2×2 Average Pooling
↓
Pooling Comparison
↓
Small Character Preservation Analysis
↓
Quantitative Analysis

---

## 📂 Project Structure

```text
pooling-document-analysis/
│
├── dataset/
│   └── sample_document.png
│
├── results/
│   ├── original.png
│   ├── grayscale.png
│   ├── preprocessed.png
│   ├── preprocessing_comparison.png
│   ├── edges.png
│   ├── edge_comparison.png
│   ├── max_pooling.png
│   ├── max_pooling_comparison.png
│   ├── average_pooling.png
│   ├── average_pooling_comparison.png
│   ├── pooling_comparison.png
│   ├── pooling_analysis.txt
│   ├── small_character_analysis.png
│   ├── small_character_metrics.txt
│   ├── final_metrics.csv
│   ├── spatial_resolution_graph.png
│   └── feature_retention_graph.png
│
├── src/
│   ├── create_dataset.py
│   ├── preprocessing.py
│   ├── edge_detection.py
│   ├── pooling_analysis.py
│   ├── average_pooling.py
│   ├── compare_pooling.py
│   ├── small_character_analysis.py
│   └── final_analysis.py
│
├── screenshots/
├── notebooks/
│
├── README.md
├── requirements.txt
└── .gitignore