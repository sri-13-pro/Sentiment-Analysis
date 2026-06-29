# ==========================================================
# Amazon Food Reviews Sentiment Analysis using VADER
# ==========================================================

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# ==========================================================
# Download Required NLTK Resources
# ==========================================================

nltk.download("punkt", quiet=True)
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords")

tqdm.pandas()

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Amazon Food Reviews Dataset...")
print("=" * 60)

df = pd.read_csv("Reviews.csv")

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Records")
print(df.head())

# ==========================================================
# Data Cleaning
# ==========================================================

print("\nCleaning Dataset...")

df = df.dropna(subset=["Text"])
df = df.reset_index(drop=True)

print("Missing values removed.")
print("Current Shape:", df.shape)

# ==========================================================
# Tokenization Example
# ==========================================================

print("\nTokenizing First Review...")

sample_text = str(df.loc[0, "Text"])

try:
    tokens = word_tokenize(sample_text)
except LookupError:
    print("Tokenizer resource not found. Using split() instead.")
    tokens = sample_text.split()

print("\nSample Tokens:")
print(tokens[:20])

# ==========================================================
# Initialize VADER
# ==========================================================

sia = SentimentIntensityAnalyzer()

# ==========================================================
# Function to Calculate Sentiment Scores
# ==========================================================

def get_sentiment_scores(text):
    return sia.polarity_scores(str(text))

# ==========================================================
# Perform Sentiment Analysis
# ==========================================================

print("\n" + "=" * 60)
print("Starting Sentiment Analysis...")
print("Please wait while all reviews are analyzed...")
print("=" * 60)

sentiment_scores = df["Text"].progress_apply(get_sentiment_scores)

print("\nSentiment Analysis Completed!")

# Convert dictionary to DataFrame
vaders = pd.DataFrame(sentiment_scores.tolist())

# Merge with original dataset
df = pd.concat([df, vaders], axis=1)

print("Sentiment Scores Added Successfully!")

# ==========================================================
# Create Sentiment Labels
# ==========================================================

def sentiment_label(compound):
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["compound"].apply(sentiment_label)

# ==========================================================
# Display Results
# ==========================================================

print("\nFirst Five Sentiment Scores")

print(df[["neg", "neu", "pos", "compound", "Sentiment"]].head())

print("\nSentiment Distribution")

counts = df["Sentiment"].value_counts()

print(counts)

print("\nSentiment Percentage")

print((counts / len(df) * 100).round(2))

# ==========================================================
# Visualization - Bar Chart
# ==========================================================

print("\nGenerating Visualizations...")

bar_counts = counts.reindex(["Positive", "Neutral", "Negative"], fill_value=0)

plt.figure(figsize=(8, 5))

bar_counts.plot(
    kind="bar",
    color=["green", "gray", "red"]
)

plt.title("Amazon Food Reviews Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

# Save Bar Chart
plt.savefig("sentiment_bar_chart.png", dpi=300)

print("✓ Bar Chart Saved")

# ==========================================================
# Visualization - Pie Chart
# ==========================================================

pie_counts = counts.reindex(["Positive", "Neutral", "Negative"], fill_value=0)

plt.figure(figsize=(7, 7))

pie_counts.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90,
    colors=["green", "gray", "red"],
    explode=(0.03, 0.03, 0.03)
)

plt.ylabel("")
plt.title("Amazon Food Reviews Sentiment Distribution")
plt.tight_layout()

# Save Pie Chart
plt.savefig("sentiment_pie_chart.png", dpi=300)

print("✓ Pie Chart Saved")

# ==========================================================
# Display Both Charts
# ==========================================================

print("Opening visualization windows...")
plt.show()

print("\n✓ All Visualizations Generated Successfully!")

# ==========================================================
# Save Output
# ==========================================================

df.to_csv("sentiment_analysis_output.csv", index=False)

print("\nOutput saved successfully!")

print("Generated Files:")
print("✓ sentiment_analysis_output.csv")
print("✓ sentiment_bar_chart.png")
print("✓ sentiment_pie_chart.png")

print("\nProject Completed Successfully!")

print("=" * 60)