import requests
from bs4 import BeautifulSoup

def fetch_news_from_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    articles = []
    for article in soup.find_all('div', class_='article'):  # Replace with actual HTML structure
        title = article.find('h2').get_text()
        content = article.find('p').get_text()
        articles.append({'title': title, 'content': content})
    
    return articles

# Example usage:
url = 'https://newsapi.org/v2/everything?q=Apple&from=2024-07-06&sortBy=popularity&apiKey=API_KEY'
articles = fetch_news_from_website(url)


    







import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained model and tokenizer
model_name = 'mrm8488/bert-tiny-finetuned-fake-news-detection'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

def predict_fake_news(text):
    if text is None:
        return "Unknown"
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(predictions, dim=1).item()
    
    # Return the prediction
    return "Fake" if predicted_label == 1 else "Real"










import streamlit as st
import requests

# Function to fetch news articles from NewsAPI
def fetch_news_from_api(api_key):
    url = 'https://newsapi.org/v2/top-headlines'
    params = {
        'apiKey': api_key,
        'country': 'us',
        'pageSize': 50  # Adjust as needed
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            articles = [{'title': article['title'], 'content': article['description']} for article in data['articles']]
            return articles
        else:
            print(f'Error fetching news: {response.text}')
            return []  # Return empty list on error
    except Exception as e:
        print(f'Exception during news fetching: {str(e)}')
        return []  # Return empty list on exception

# Streamlit application
st.title("Real-Time Fake News Detection with BERT")

def fetch_and_display_news():
    st.subheader("Latest News Articles")
    
    api_key = '484aad51e72d46d9b450332fbfbd38c3'  # Replace with your actual API key
    articles = fetch_news_from_api(api_key)
    
    if articles:
        for article in articles:
            st.write(f"**Title:** {article['title']}")
            content = article.get('content')
            st.write(f"**Content:** {content}")
            prediction = predict_fake_news(content)
            st.write(f"**Prediction:** {prediction}")
            st.write("---")
    else:
        st.write("No articles fetched. Check your API key or website scraping function.")

if st.button("Fetch News"):
    fetch_and_display_news()