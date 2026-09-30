# Simple RNN Text Classification

This project is a small deep learning example that classifies short text samples as either positive or negative using a recurrent neural network implemented with TensorFlow/Keras.

The code in `simpleRNN.py` demonstrates:

- text preprocessing with `Tokenizer`
- sequence padding with `pad_sequences`
- train/test splitting
- an embedding layer plus LSTM model
- model training and evaluation
- prediction on a new sample sentence

## Project files

- `simpleRNN.py` — main script containing the training and prediction pipeline
- `main.py` — minimal placeholder script
- `pyproject.toml` — project metadata

## Model overview

The model uses a binary classification setup for sentiment-style text labels:

- `Embedding(input_dim=1000, output_dim=16, input_length=max_len)`
- `LSTM(32)`
- `Dense(16, activation='relu')`
- `Dense(1, activation='sigmoid')`

It is compiled with:

- loss: `binary_crossentropy`
- optimizer: `adam`
- metric: `accuracy`

The script includes a comment indicating that the `LSTM` layer can be replaced with `SimpleRNN` or `GRU`, depending on the architecture you want to experiment with.

## Data

The sample dataset contains 10 short text examples with labels:

- `1` = positive / relevant text
- `0` = negative / less relevant text

Example inputs:

- "I love machine learning"
- "AI is the future"
- "I dislike bad predictions"
- "Sometimes models overfit"

## Requirements

Install the Python dependencies before running the example:

```bash
pip install tensorflow numpy scikit-learn
```

You may also use a virtual environment for isolation.

## Run the project

From the project folder, run:

```bash
python simpleRNN.py
```

This will:

1. tokenize and pad the sample texts
2. split the data into training and testing sets
3. train the LSTM model for 10 epochs
4. print the test accuracy
5. predict the class of a sample input such as "I love AI"

## Example output

You should see output similar to:

```text
Test Accuracy: 0.5000
Prediction (probability): 0.8461
Class: Positive
```

The exact numbers may vary depending on the training run and environment.

## Notes

- This is a beginner-friendly example intended for learning how text classification works with RNNs.
- The dataset is intentionally small and simple.
- For real-world tasks, use a larger dataset and a more robust model architecture.

## Future improvements

Possible extensions include:

- using a larger text dataset
- trying `SimpleRNN` and `GRU` layers
- adding validation and early stopping
- improving preprocessing with stopword removal and stemming
- evaluating with a confusion matrix and classification report

## License

This project is provided for educational purposes.
