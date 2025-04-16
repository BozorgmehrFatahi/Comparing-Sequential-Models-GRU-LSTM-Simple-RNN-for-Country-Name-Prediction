In this project, using a dataset of names of people in different countries, we intend to predict which country a name belongs to. This project has been modeled separately with three different sequential models, including Simple RNN, LSTM, and GRU, and the results of the modeling have been compared with each other.

# RNN Model Comparison: GRU vs LSTM vs SimpleRNN for Name Origin Classification

This project presents a comparative study of three popular Recurrent Neural Network (RNN) architectures—**GRU**, **LSTM**, and **SimpleRNN**—applied to a name classification task. The goal is to predict the **origin** of a given name based on its character-level features using deep learning.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset Description](#dataset-description)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures](#model-architectures)
- [Training Configuration](#training-configuration)
- [Evaluation and Comparison](#evaluation-and-comparison)
- [Result Visualization](#result-visualization)
- [Conclusion](#conclusion)

---

## Overview

The script:
1. Mounts Google Drive to access the dataset.
2. Loads names grouped by origin from `.txt` files.
3. Encodes names into one-hot character vectors.
4. Trains three different RNN models:
   - GRU-based
   - LSTM-based
   - SimpleRNN-based
5. Visualizes and compares training/validation losses using `matplotlib`.

---

## Requirements

Make sure you have the following libraries installed (Google Colab comes with most pre-installed):

```bash
numpy
pandas
matplotlib
tensorflow / keras
sklearn
```

---

## Dataset Description

The dataset is organized in your Google Drive under:

```
/MyDrive/train/
```

Each `.txt` file represents one class (i.e., name origin) and contains a list of names. For example:

```
/train/English.txt
/train/Japanese.txt
/train/Arabic.txt
```

Each line inside these files is one name.

---

## Data Preprocessing

- All names are loaded into a DataFrame with two columns: `Origin` and `Name`.
- The character set is extracted from all unique characters found in the names.
- Names are encoded as 2D one-hot matrices of shape `(max_length, num_characters)`.
- The target (origin) is one-hot encoded using `pandas.get_dummies`.
- The dataset is split into training and validation sets (90%/10%).

---

## Model Architectures

All three models share a common structure:
- A single RNN layer (either GRU, LSTM, or SimpleRNN) with 256 units.
- A fully connected output layer with sigmoid activation for multi-class classification.

Example (GRU):
```python
model = Sequential()
model.add(GRU(units=256, input_shape=(max_length, num_characters)))
model.add(Dense(units=num_classes, activation='sigmoid'))
```

Similar structures are used for LSTM and SimpleRNN by swapping the recurrent layer type.

---

## Training Configuration

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: AUC (Area Under the Curve)
- **Epochs**: 120
- **Batch Size**: 1024

Training is conducted independently for each of the three models using the same configuration.

---

## Evaluation and Comparison

After training, each model’s performance is plotted using `matplotlib`, showing both training and validation loss over epochs.

### Observations:
- **GRU**: Fast convergence and stable validation performance. Good balance between speed and accuracy.
- **LSTM**: Slightly more training time than GRU but better at capturing long-term dependencies.
- **SimpleRNN**: Lightweight and fast, but less effective on complex sequences; tends to overfit or underperform compared to GRU and LSTM.

---

## Result Visualization

Each model produces a training curve showing how the loss evolves over time.

In each plot:
- **Red** line: Training loss
- **Blue** line: Validation loss

These graphs are shown separately after training each model, helping visualize learning behavior and generalization.

You can modify the code to save each plot as an image file:

```python
plt.savefig('gru_loss_curve.png')
```

---

## Conclusion

This experiment demonstrates that:
- GRU and LSTM both outperform SimpleRNN in this classification task.
- LSTM is slightly more robust but computationally heavier.
- GRU offers a good trade-off between complexity and performance.
- SimpleRNN can serve as a baseline but may struggle with longer sequences.

For best results in sequence modeling tasks such as name classification, **GRU** or **LSTM** is recommended.

---

## Future Work

- Add more layers or use bidirectional RNNs.
- Include dropout or batch normalization for regularization.
- Evaluate models using additional metrics like accuracy and confusion matrix.
- Experiment with attention mechanisms or transformers.

---
