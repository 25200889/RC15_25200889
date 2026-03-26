import pandas as pd
from textblob import TextBlob

# 1. Read CSV file
df = pd.read_csv('P.4--Text.csv', encoding='utf-8')

# 2. Sentiment analysis function
def get_sentiment(text):
    if pd.isna(text) or str(text).strip() == '':
        return 0.0
    blob = TextBlob(str(text))
    return blob.sentiment.polarity

df['polarity'] = df['text'].apply(get_sentiment)

# 3. Classify sentiment category
def classify(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['polarity'].apply(classify)

# 4. Save result CSV (including required columns)
output_cols = ['index', 'start_time', 'end_time', 'text', 'polarity', 'sentiment']
df[output_cols].to_csv('sentiment_analysis_results.csv', index=False, encoding='utf-8')
print("Sentiment analysis results saved to sentiment_analysis_results.csv")

# 5. Output statistics
max_p = df['polarity'].max()
min_p = df['polarity'].min()
mean_p = df['polarity'].mean()
print("\n=== Sentiment Polarity Statistics ===")
print(f"Maximum score: {max_p:.4f}")
print(f"Minimum score: {min_p:.4f}")
print(f"Average score: {mean_p:.4f}")

# Optional: output category counts
print("\nSentiment Category Distribution:")
print(df['sentiment'].value_counts())