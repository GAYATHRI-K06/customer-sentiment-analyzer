import pandas as pd
df=pd.read_csv("C:\\Users\\admin\\OneDrive\\Documents\\sentiment\\data\\Reviews.csv")
print('\nShape\n',df.shape)
print('\nColums\n',df.columns.tolist())
print('\nFirst 5 rows\n',df.head())
print('\n Missing values\n',df.isnull().sum())

df=df[['Score','Text']]
df=df.dropna()
print('\n cleaned shape:\n',df.shape)
print('\nScore distribution\n',df['Score'].value_counts())

def lable_sentiment(Score):
    if Score>=4:
        return 'positive'
    elif Score==3:
        return 'neutral'
    else:
        return 'negative'
    
df['sentiment']=df['Score'].apply(lable_sentiment)
print(df['sentiment'].value_counts())
# Take a sample of 10,000 rows
df = df.sample(10000, random_state=42)
print("Sampled shape:", df.shape)

from textblob import TextBlob
def get_sentiment(text):
    analyse=TextBlob(str(text))
    polarity=analyse.sentiment.polarity
    if polarity >0:
        return 'Positive'
    elif polarity ==0:
        return 'Neutral'
    else:
        return 'Negative'
df['Text']=df['Text'].str.replace('<br />', ' ')    
df['TextBlob_sentiment']=df['Text'].apply(get_sentiment)
print(df['TextBlob_sentiment'].value_counts())
print(df[['Text','sentiment','TextBlob_sentiment']].head())
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,5))
sns.countplot(x='TextBlob_sentiment',data=df,palette=['red','gray','green'],order=['Negative','Neutral','Positive'])
plt.title("Customer Sentiment Distribution")
plt.xlabel("sentiment")
plt.ylabel("No of reviews")
plt.savefig("sentiment_distribution.png")
plt.close()

print("Chart 1 saved")

plt.figure(figsize=(8,5))
df.groupby('TextBlob_sentiment')['Score'].mean().plot(kind='bar', 
                                                       color=['red','gray','green'])
plt.title("Average Score by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Average Score")
plt.savefig("average_score_by_sentiment.png")
plt.close()
print("Chart 2 saved")

# Chart 3 - Comparing star rating sentiment vs TextBlob sentiment
comparison = pd.crosstab(df['sentiment'], df['TextBlob_sentiment'])
print(comparison)

plt.figure(figsize=(8,5))
comparison.plot(kind='bar', colormap='RdYlGn')
plt.title("Star Rating Sentiment vs TextBlob Sentiment")
plt.xlabel("Star Rating Sentiment")
plt.ylabel("Count")
plt.legend(title="TextBlob Sentiment")
plt.savefig("charts/comparison.png")
plt.close()
print("Chart 3 saved!")

from wordcloud import WordCloud

# Word cloud for positive reviews
positive_text = ' '.join(df[df['TextBlob_sentiment'] == 'Positive']['Text'])

wordcloud = WordCloud(width=800, height=400, 
                      background_color='white',
                      colormap='Greens').generate(positive_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Words in Positive Reviews")
plt.savefig("charts/wordcloud_positive.png")
plt.close()
print("Word cloud saved!")