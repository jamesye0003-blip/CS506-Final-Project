# Urban Sound Classification

##  Project Overview 

This project aims to build a machine learning pipeline to classify 10+ categories of urban environmental sounds, such as traffic noise, sirens, birdsong, and construction noise, enabling smart-city monitoring and analysis of acoustic environments.

## Goals 

**Accurate Sound Recognition**: Build a model that can label environmental sounds with high accuracy.

## Data Collection 

[UrbanSound8K dataset](https://audeering.github.io/datasets/datasets/urbansound8k.html)  
Contains 8732 labeled sound clips (<=4s) across 10 urban classes.  

**Optional**: Sharing the final cleaned dataset on Kaggle or Hugging Face for reproducibility.

## Modeling

### **Feature Extraction**

**Basic method**: Compute Mel-spectrograms.
**Optional**:  Apply data augmentation (time stretching, pitch shifting).

### **Model Architectures**

**Baseline**: 2D CNN with batch norm and dropout.

**Optional**: Transfer learning using a pretrained audio model.

**Comparative Models**:

- Transfer learning using pretrained audio models (e.g., VGGish, YAMNet).
- AudioCLIP for inference-level benchmarking against the baseline.


## Visualization

**Basic visualization method**: Interactive confusion matrix.

**Additional planned visualizations**: 

- Precision-recall curves.
- t-SNE or PCA projections of learned representations.

- Per-class accuracy plots.


## Test Plan

**Split**: 80/20 train/test, stratified by class.

**Cross-validation**: 5-fold on training set to tune hyperparameters.

Compare performance across models and feature representations.