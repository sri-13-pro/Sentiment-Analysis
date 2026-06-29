# 🍽️ Amazon Food Reviews Sentiment Analysis

## 📌 Overview

This project performs **Sentiment Analysis** on Amazon Fine Food Reviews using **Natural Language Processing (NLP)** techniques. The application classifies customer reviews into **Positive**, **Negative**, or **Neutral** categories using the **VADER (Valence Aware Dictionary and sEntiment Reasoner)** sentiment analysis model from the NLTK library.

The project also generates visualizations to help understand the overall sentiment distribution and exports the analyzed results to a CSV file.

---

## 🚀 Features

* 📂 Load and process the Amazon Fine Food Reviews dataset
* 🧹 Clean and preprocess review text
* 🔤 Tokenize review text
* 😊 Perform sentiment analysis using VADER
* 📊 Generate sentiment labels (Positive, Neutral, Negative)
* 📈 Create bar chart visualization
* 🥧 Create pie chart visualization
* 💾 Export processed data to CSV
* ⏳ Progress bar for large datasets using `tqdm`

---

## 📂 Dataset

This project uses the **Amazon Fine Food Reviews** dataset.

* **Source:** Kaggle
* **Reviews:** 568,454
* **Format:** CSV

Download the dataset:

https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

Place the downloaded `Reviews.csv` file inside the project folder.

---

## 🛠️ Technologies Used

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Seaborn
* NLTK
* tqdm

---

## 📦 Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/amazon-food-sentiment-analysis.git
```

```bash
cd amazon-food-sentiment-analysis
```

---

### Create Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn nltk tqdm
```

---

## ▶️ Run the Project

```bash
python main.py
```

---

## 📊 Output

The program performs the following steps:

1. Loads the dataset
2. Cleans missing values
3. Tokenizes sample review
4. Calculates sentiment scores
5. Classifies reviews
6. Displays sentiment statistics
7. Generates visualizations
8. Saves the final dataset

---

## 📈 Visualizations

The project generates:

* Sentiment Distribution Bar Chart
* Sentiment Distribution Pie Chart

These charts help understand the overall customer opinion about Amazon food products.

---

## 📁 Project Structure

```text
Amazon-Food-Sentiment-Analysis/
│Sentiment-Analysis/
## 📂 Project Structure

```text
Amazon-Food-Sentiment-Analysis/
│
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
├── README.md                   # Project documentation
├── main.py                     # Main sentiment analysis script
├── requirements.txt            # Python dependencies
├── sentiment_bar_chart.png     # Generated bar chart
├── sentiment_pie_chart.png     # Generated pie chart
└── sentiment_analysis_output.csv   # Generated after running the project
```
---

## 🧠 Sentiment Classification

The VADER compound score is used for classification.

| Compound Score         | Sentiment |
| ---------------------- | --------- |
| >= 0.05                | Positive  |
| <= -0.05               | Negative  |
| Between -0.05 and 0.05 | Neutral   |

---

## 📊 Sample Output

```
Dataset Shape: (568454, 10)

Positive : 392345
Neutral  : 87321
Negative : 88842

Project Completed Successfully!
```

---

## 🔮 Future Improvements

* Word Cloud visualization
* Interactive dashboard using Streamlit
* Deep Learning sentiment analysis (LSTM/BERT)
* Real-time Amazon review prediction
* Confusion matrix and model evaluation
* Web application deployment

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Srinath Rajasekar R**

Data Analyst | Machine Learning Enthusiast

GitHub: https://github.com/sri-13-pro

LinkedIn:https://www.linkedin.com/in/srinath-rajasekar-r-10b52b339

---

⭐ If you found this project helpful, consider giving it a star on GitHub.
