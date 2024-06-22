import streamlit as st
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import requests


def main(date):
        # Load the data from Excel files
    st.write("Find the summary below for  {0}".format(selected_option))
    file_path_indices = "MW-All-Indices-21-Jun-2024.csv"
    file_path_nifty = "MW-NIFTY-50-21-Jun-2024.csv"


    # In[3]:


    indices_data = pd.read_csv(file_path_indices)
    nifty_data = pd.read_csv(file_path_nifty)


    # In[4]:


    # Clean and preprocess the data
    indices_data.columns = ['INDEX', 'CURRENT', '%CHNG', 'OPEN', 'HIGH', 'LOW', 'INDICATIVE CLOSE', 'PREV. CLOSE', 'PREV. DAY', 
                            '1W AGO', '1M AGO', '1Y AGO', '52W HIGH', '52W LOW', '365D % CHNG', '30D % CHNG']
    nifty_data.columns = ['SYMBOL', 'OPEN', 'HIGH', 'LOW', 'PREV. CLOSE', 'LTP', 'CHNG', '%CHNG', 'VOLUME', 'VALUE', 
                        '52W HIGH', '52W LOW', '30D %CHNG', '365D %CHNG']


    # In[5]:


    indices_data.columns = indices_data.columns.str.strip()
    nifty_data.columns = nifty_data.columns.str.strip()


    # In[6]:


    def convert_to_numeric(df, columns):
        for col in columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')



    # In[12]:


    # Convert relevant columns to numeric in both datasets
    convert_to_numeric(indices_data, ['CURRENT', '%CHNG', 'OPEN', 'HIGH', 'LOW', 'PREV. CLOSE', 'PREV. DAY', '1W AGO', '1M AGO', '1Y AGO', '52W HIGH', '52W LOW', '365D % CHNG', '30D % CHNG'])
    convert_to_numeric(nifty_data, ['OPEN', 'HIGH', 'LOW', 'PREV. CLOSE', 'LTP', 'CHNG', '%CHNG', 'VOLUME', 'VALUE', '52W HIGH', '52W LOW', '30D %CHNG', '365D %CHNG'])


    # In[13]:


    # Define labels based on percentage change
    def categorize_change(change):
        if change > 0:
            return 'Positive'
        elif change < 0:
            return 'Negative'
        else:
            return 'Neutral'


    # In[14]:


    indices_data['LABEL'] = indices_data['%CHNG'].apply(categorize_change)
    nifty_data['LABEL'] = nifty_data['%CHNG'].apply(categorize_change)


    # In[15]:


    combined_data = pd.concat([indices_data, nifty_data], ignore_index=True)


    # In[16]:


    features = combined_data[['CURRENT', '%CHNG', 'OPEN', 'HIGH', 'LOW', 'PREV. CLOSE', '1W AGO', '1M AGO', '1Y AGO', '52W HIGH', '52W LOW', '30D % CHNG', '365D % CHNG']]
    labels = combined_data['LABEL']


    # In[17]:


    # Drop non-numeric columns from the features
    features_numeric = features.select_dtypes(include=[float, int])


    # In[18]:


    # Create an imputer object with a median filling strategy
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(features_numeric)


    # In[19]:


    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, labels, test_size=0.2, random_state=42)


    # In[20]:


    # Initialize and train the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)


    # In[21]:


    # Fetch real financial news articles using NewsAPI
    api_key = '7c67d778f3124da4a7edc25cac4bb7ba'
    url = f'https://newsapi.org/v2/everything?q=finance&apiKey={api_key}'
    response = requests.get(url)
    data = response.json()
    articles = data['articles']


    # In[22]:


    # Extract articles with handling None values
    news_articles = []
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        if title and description:
            news_articles.append(title + ". " + description)
        elif title:
            news_articles.append(title)
        elif description:
            news_articles.append(description)


    # In[23]:


    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()


    # In[24]:


    def analyze_sentiment(text):
        sentiment_score = analyzer.polarity_scores(text)
        return sentiment_score


    # In[25]:


    sentiment_scores = [analyze_sentiment(article) for article in news_articles]


    # In[26]:


    # Topic Modeling
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(news_articles)

    lda = LatentDirichletAllocation(n_components=3, random_state=0)
    lda.fit(dtm)


    # In[27]:


    def display_topics(model, feature_names, no_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        return topics


    # In[28]:


    topics = display_topics(lda, vectorizer.get_feature_names_out(), 5)


    # In[29]:


    # Generate comprehensive summary for all indices
    def generate_comprehensive_summary(data, sentiment_scores, topics):
        total_indices = len(data)
        positive_count = len(data[data['LABEL'] == 'Positive'])
        negative_count = len(data[data['LABEL'] == 'Negative'])
        neutral_count = len(data[data['LABEL'] == 'Neutral'])
        
        most_positive = data[data['%CHNG'] == data['%CHNG'].max()]
        most_negative = data[data['%CHNG'] == data['%CHNG'].min()]
        
        avg_change = data['%CHNG'].mean()
        
        summary = (f"Today's market summary:\n\n"
                f"Out of {total_indices} indices, {positive_count} ended in positive territory, "
                f"{negative_count} in negative territory, and {neutral_count} remained unchanged.\n\n"
                f"The index with the most significant gain was {most_positive.iloc[0]['INDEX']} "
                f"with a rise of {most_positive.iloc[0]['%CHNG']:.2f}% to {most_positive.iloc[0]['CURRENT']:.2f} points.\n"
                f"The index with the most significant loss was {most_negative.iloc[0]['INDEX']} "
                f"with a drop of {most_negative.iloc[0]['%CHNG']:.2f}% to {most_negative.iloc[0]['CURRENT']:.2f} points.\n\n"
                f"The average change across all indices was {avg_change:.2f}%.\n\n"
                f"Sentiment Analysis of Recent News:\n")
        
        for score in sentiment_scores:
            summary += f" - Positive: {score['pos']:.2f}, Negative: {score['neg']:.2f}, Neutral: {score['neu']:.2f}\n"
        
        summary += "\nKey Topics in Recent News:\n"
        for topic in topics:
            summary += f" - {topic}\n"
        
        return summary


    # In[30]:


    # Generate comprehensive summary for Nifty 50
    def generate_nifty_summary(data, sentiment_scores, topics):
        total_indices = len(data)
        positive_count = len(data[data['LABEL'] == 'Positive'])
        negative_count = len(data[data['LABEL'] == 'Negative'])
        neutral_count = len(data[data['LABEL'] == 'Neutral'])
        
        most_positive = data[data['%CHNG'] == data['%CHNG'].max()]
        most_negative = data[data['%CHNG'] == data['%CHNG'].min()]
        
        avg_change = data['%CHNG'].mean()
        
        summary = (f"Today's Nifty 50 summary:\n\n"
                f"Out of {total_indices} stocks, {positive_count} ended in positive territory, "
                f"{negative_count} in negative territory, and {neutral_count} remained unchanged.\n\n"
                f"The stock with the most significant gain was {most_positive.iloc[0]['SYMBOL']} "
                f"with a rise of {most_positive.iloc[0]['%CHNG']:.2f}% to {most_positive.iloc[0]['LTP']:.2f} points.\n"
                f"The stock with the most significant loss was {most_negative.iloc[0]['SYMBOL']} "
                f"with a drop of {most_negative.iloc[0]['%CHNG']:.2f}% to {most_negative.iloc[0]['LTP']:.2f} points.\n\n"
                f"The average change across all stocks was {avg_change:.2f}%.\n\n"
                f"Sentiment Analysis of Recent News:\n")
        
        for score in sentiment_scores:
            summary += f" - Positive: {score['pos']:.2f}, Negative: {score['neg']:.2f}, Neutral: {score['neu']:.2f}\n"
        
        summary += "\nKey Topics in Recent News:\n"
        for topic in topics:
            summary += f" - {topic}\n"
        
        return summary


    # In[31]:


    # Generate and print the comprehensive summary
    final_summary = generate_comprehensive_summary(combined_data, sentiment_scores, topics)
    nifty_summary = generate_nifty_summary(nifty_data, sentiment_scores, topics)


    # In[32]:


    st.write(final_summary)
    # print("\n")






# Title of the app
st.title("Select The Date for News Flash")


startDate = datetime(2024, 6,21)
datestring=startDate.strftime("%Y-%m-%d")
# Options for the dropdown menu
options = ["select from below",datestring]

# Creating the dropdown menu
selected_option = st.selectbox("Choose an option:", options)

# Display the selected option
# st.write("You selected:", selected_option)

if selected_option != "select from below":
    main(startDate)
else:
    st.write("Please select an option from the dropdown menu.")
