# Volleyball Neural Network

This repository contains a TensorFlow/Keras implementation for predicting the winner of a volleyball match based on statistical differences between two teams. The dataset should be stored in a CSV file named `vnl_dataset.csv` in the repository root or provide the path accordingly.

## Usage

1. Install the required dependencies:
   ```bash
   pip install pandas scikit-learn tensorflow matplotlib
   ```
2. Run the training script:
   ```bash
   python train_volleyball_nn.py
   ```
   The script loads the dataset, splits it into training and test sets, trains a neural network, prints evaluation metrics, and optionally plots training curves and a ROC curve.

## Dataset Format

The CSV file must contain the following columns:
- `attack_diff`
- `block_diff`
- `serve_diff`
- `opp_error_diff`
- `total_points_diff`
- `dig_diff`
- `reception_diff`
- `set_diff`
- `top_scorer_1_diff`
- `top_scorer_2_diff`
- `label` (1 if Team A wins, 0 if Team B wins)

Ensure there are no missing values for these columns.
