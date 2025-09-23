# Urban Sound Classification

##  Project Overview 

This project aims to build a machine learning pipeline to classify 10+ categories of urban environmental sounds, such as traffic noise, sirens, birdsong, and construction noise, enabling smart-city monitoring and analysis of acoustic environments.

## Goals 

Accurate Sound Recognition: Build a model that can label environmental sounds with high accuracy.

## Data Collection 

[UrbanSound8K dataset](https://audeering.github.io/datasets/datasets/urbansound8k.html)  
Contains 8732 labeled sound clips (<=4s) across 10 urban classes.  

## Modeling

**Feature Extraction**:
Compute Mel-spectrograms.
Optional:  apply data augmentation (time stretching, pitch shifting).

**Model Architectures**

Baseline: 2D CNN with batch norm and dropout.

Optional: Transfer learning using a pretrained audio model.

## Visualization

Interactive confusion matrix.

## Test Plan

Split: 80/20 train/test, stratified by class.

Cross-validation: 5-fold on training set to tune hyperparameters.
