# Twitter Sentiment Analysis using LSTM (Refined)

This project performs **sentiment analysis on tweets** using a **neural network (LSTM)** enhanced with **pre-trained GloVe embeddings**, **custom preprocessing**, and **early stopping**.  It can classify user-input text as **Positive**, **Negative**, or **Neutral**.

---

## 🚀 Overview

This project refines and extends the Kaggle notebook:

### 👨‍💻 Author

> **Original work:** [EDA Twitter Sentiment Analysis using NN](https://www.kaggle.com/code/muhammadimran112233/eda-twitter-sentiment-analysis-using-nn)  
> **Original Author:** Muhammad Imran

> **Modified by:** [Silas](https://github.com/SilasVerse)

> (for academic purpose)

> All preprocessing, model structure, and evaluation were revisited for better accuracy and interpretability.

---

## ✨ Improvements over Original Version

| Area | Original | Refined Version |
|------|-----------|----------------|
| **Embeddings** | Random initialization | Pre-trained **GloVe Twitter 100d** embeddings |
| **Stopwords** | Default NLTK | Customized (keeps *no, not, nor* for negation handling) |
| **Cleaning** | Basic lowercase + punctuation removal | Added URL, email, digit, and repeat-character cleaning |
| **Tokenizer** | `num_words=2000` | Increased to `num_words=10000` for richer vocabulary |
| **Training** | Fixed 10 epochs | Used **EarlyStopping** with validation monitoring |
| **Model** | Basic LSTM (64 units) | Optimized architecture with dropout and regularization |
| **Accuracy** | ~0.57 | Improved to **~0.72–0.80** |
| **Prediction Interface** | VADER-only | LSTM-based + **VADER fallback** for uncertain predictions |
| **User Interaction** | Static dataset only | Added real-time text input prediction |

---

## 🧩 Model Architecture

Input (max_len = 500)

   ↓
   
Embedding (GloVe pretrained vectors)

   ↓
   
LSTM (64 units)

   ↓
   
Dense (256, ReLU)

   ↓
   
Dropout (0.5)

   ↓
   
Dense (1, Sigmoid)

--------------

## Download dataset

From: [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

Use the dataset:

"training.1600000.processed.noemoticon.csv"

----
## Download GloVe embeddings

From: [GloVe Twitter 100d](https://nlp.stanford.edu/projects/glove/)

Then extract "glove.twitter.27B.100d.txt"

-------------

## ⚙️ Technologies Used

- Python 3.x

- TensorFlow / Keras

- NLTK

- Pandas / NumPy

- Matplotlib

- GloVe pretrained embeddings

- VADER Sentiment Analyzer (fallback)

-----
## 📚 Credits

- [Kaggle Notebook](https://www.kaggle.com/code/muhammadimran112233/eda-twitter-sentiment-analysis-using-nn)

- [Stanford NLP GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)

- [NLTK VADER Lexicon](https://github.com/cjhutto/vaderSentiment)

------

## 🪪 License

This project is open source under the MIT License.
Please see the [LICENSE](https://github.com/SilasVerse/Twitter-Sentiment-Analysis-Upgrade/blob/main/LICENSE)

Note: This repository contains significant modifications and enhancements to the original Kaggle notebook.
Proper attribution is maintained in accordance with open-source ethics.

---------
### 👨‍💻 Author

- Silas

- 📧 silasanderson.rpce@gmail.com

- 🌐 [SilasVerse](https://github.com/SilasVerse) 
