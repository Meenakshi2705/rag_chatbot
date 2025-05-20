# 🎬 IMDBot: AI-powered Movie Information & Recommendation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

IMDBot is a conversational Streamlit application that answers movie-related queries and suggests recommendations using semantic search, sentiment analysis, and metadata filtering. Just upload an IMDB-style CSV and start chatting!

---
## 🖼️ Screenshots

### 🔹 Home Page
![image](https://github.com/user-attachments/assets/d7c30bb4-ed6c-477e-9ea9-b4f2b5e28fa6)

## 🚀 Features

- 📊 **Semantic Movie Search** – Ask for top, best, or recommended movies across genres.
- 🤖 **NLP-Driven Recommendations** – Uses sentence embeddings and FAISS to find relevant matches.
- ❤️ **Sentiment-Aware Ranking** – Analyzes plot sentiment to influence movie rankings.
- 🧠 **Attribute-based Q&A** – Get info like rating, director, runtime, revenue, and more.
- 🔍 **Genre & Rating Filters** – Filter recommendations by genre or rating thresholds.
- 📂 **Custom Dataset Support** – Upload your own IMDB-format CSV file.

---

## 🧠 Tech Stack

- [Streamlit](https://streamlit.io/) – Web interface
- [FAISS](https://github.com/facebookresearch/faiss) – Fast similarity search
- [Sentence Transformers](https://www.sbert.net/) – Semantic embeddings
- [TextBlob](https://textblob.readthedocs.io/en/dev/) – Sentiment analysis
- [LangChain CSVLoader](https://docs.langchain.com/docs/integrations/document_loaders/csv) – CSV ingestion

---

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/imdbot.git
cd imdbot
pip install -r requirements.txt
streamlit run app.py


