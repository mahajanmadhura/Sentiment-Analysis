import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def visualize_data(df):
    print("\n=== Data Visualization ===")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='sentiment', data=df)
    plt.title('Sentiment Distribution')
    plt.show()
    for sentiment in df['sentiment'].unique():
        text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud: {sentiment.capitalize()} Sentiment')
        plt.axis('off')
        plt.show()
