# Adversarial Machine Learning in Cybersecurity: Evaluating Black-Box Transferability in Network Intrusion Detection Systems


# Black-Box Adversarial Transferability Experiments in NIDS

This repository contains the experimental code used for the master thesis:

**Adversarial AI in Cyber Attacks**  
Master in Applied Computer and Information Technology  
Specialization in Cybersecurity  
Oslo Metropolitan University

The experiments investigate black-box adversarial transferability in machine-learning-based Network Intrusion Detection Systems (NIDS). The main objective is to evaluate how adversarial samples generated using surrogate models transfer to different target models under realistic black-box assumptions.

## Overview

The repository includes five experiments:

1. **Experiment 1: Same-Dataset Cross-Model Transferability**  
   Evaluates whether adversarial samples generated using a Logistic Regression surrogate can transfer to different target models trained on the same dataset.

2. **Experiment 2: Cross-Dataset Transferability**  
   Tests whether adversarial samples generated from one CIC-IDS-2017 subset can transfer to models evaluated on another dataset subset.

3. **Experiment 3: Effect of Limited Training Data**  
   Investigates whether a surrogate model trained on only a limited percentage of the available data can still generate effective adversarial samples.

4. **Experiment 4: Alternative Surrogate Model**  
   Evaluates whether a Decision Tree surrogate can generate transferable adversarial samples compared with a Logistic Regression surrogate.

5. **Experiment 5: Ensemble Robustness**  
   Tests whether an ensemble-based NIDS using majority voting improves robustness against transfer-based adversarial attacks.

## Dataset

The experiments use subsets of the **CIC-IDS-2017** dataset:

- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`

The dataset is not included in this repository due to file size and licensing considerations. It can be downloaded from the official CIC-IDS-2017 dataset source.

After downloading the dataset, place the required CSV files in the same directory as the experiment scripts, or update the file paths in the code.

## Models Used

The experiments use the following machine learning models:

- Logistic Regression
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- Decision Tree
- Ensemble model using majority voting

## Evaluation Metrics

The experiments evaluate model performance using:

- Clean accuracy
- Adversarial accuracy
- Accuracy drop
- Evasion rate

These metrics are used to measure both general model performance and the effectiveness of adversarial attacks.

## Requirements

The experiments were developed using Python and common machine learning libraries.

Install the required packages using:

```bash
pip install pandas numpy scikit-learn matplotlib
