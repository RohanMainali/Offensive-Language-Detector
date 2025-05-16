# 🛡️ Offensive Content Detection with Contextual Understanding

This project leverages transformer-based language models (like BERT and T5) to detect offensive content in text, enhanced with contextual understanding and suggestions for alternative phrasing.

## 🚀 Features

* Offensive content detection using HuggingFace Transformers.
* Context-aware suggestions using T5 generation model.
* Preprocessing pipeline including tokenization, stopword removal, and regular expressions.
* Evaluation metrics including accuracy, precision, recall, and F1-score.
* GPU compatibility with PyTorch.

## 🧠 Model Architecture

* **Classification**: Uses `AutoModelForSequenceClassification` for binary/multi-label offensive detection.
* **Suggestion Generation**: Implements `T5ForConditionalGeneration` to rewrite offensive content into non-offensive alternatives.
* **Tokenizer**: Uses `AutoTokenizer` and `T5Tokenizer` for encoding text inputs.

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/offensive-content-detector.git
cd offensive-content-detector
pip install -r requirements.txt
```

**Required libraries include**:

* `transformers`
* `torch`
* `sklearn`
* `nltk`
* `pandas`, `numpy`
* `matplotlib`, `seaborn`

## 📊 Dataset

* The model expects a CSV-formatted dataset with at least:

  * `text`: the input sentences
  * `label`: binary or multi-class labels indicating offensiveness

You may modify the dataset loading and splitting sections to fit your own format.

## 🧪 Evaluation

* Uses `train_test_split` to divide data
* Evaluates with:

  * `accuracy_score`
  * `precision_recall_fscore_support`
  * `confusion_matrix`

## 📈 Output

* Model predictions
* Classification report
* Confusion matrix heatmaps
* Optional: Suggestive rewrites of flagged content

## 🖥️ Run

```python
python offensivesortafinal.py
```

Ensure you have GPU support if running large datasets.

## 📁 File Structure

```
.
├── offensivesortafinal.py      # Main script
├── data/                       # Folder for input dataset
├── models/                     # Optional: save checkpoints here
├── outputs/                    # Evaluation results and visualizations
└── README.md
```

