import streamlit as st
import pandas as pd
import numpy as np

import io
import PIL
from PIL import Image
import requests
from io import BytesIO

import difflib
import nltk
import string
import warnings
from scipy.stats import pearsonr
from nltk.corpus import stopwords
#from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
nltk.download('stopwords')
warnings.filterwarnings('ignore')

#st.set_page_config(layout = "wide", page_title='Radhika_1917631')

from PIL import Image
# Loading Image using PIL
im = Image.open('ilogo.png')
# Adding Image to web app
st.set_page_config(page_title="Radhika", page_icon = im)
st.title("TED Talks Recommendation System")
import streamlit.components.v1 as components

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
	
load_css('style.css')
from googleapiclient.discovery import build
import pandas as pd

api_key= "AIzaSyCjPkeYqEA4FqalBl9blpFs9Fnucp3kBUY"
#channel_id = "UCsT0YIqwnpJCM-mx7-gSA4Q"

channel_ids = ["UCsT0YIqwnpJCM-mx7-gSA4Q",
               "UCAuUUnT6oDeKwE6v1NGQxug",
               "UCsooa4yRKGN_zEE8iknghZA",
               "UC-yTB2bUcin9mmah36sXiYA"]        

youtube = build('youtube', 'v3', developerKey=api_key)

def get_channel_stats(youtube, channel_ids):
    all_data = []
    request = youtube.channels().list(
                part='snippet,contentDetails,statistics',
                id=','.join(channel_ids))
    response = request.execute() 
    
    for i in range(len(response['items'])):
        data = dict(Channel_name = response['items'][i]['snippet']['title'],
                    Subscribers = response['items'][i]['statistics']['subscriberCount'],
                    Views = response['items'][i]['statistics']['viewCount'],
                    Total_videos = response['items'][i]['statistics']['videoCount'],
                    playlist_id = response['items'][i]['contentDetails']['relatedPlaylists']['uploads'])
        all_data.append(data)
    
    return all_data


channel_statistics = get_channel_stats(youtube, channel_ids)
channel_data = pd.DataFrame(channel_statistics)
st.write(channel_data)

st.write('<p style="font-size:130%">Select TED talk Channel</p>', unsafe_allow_html=True)
file_data = st.radio('Channels List:', ('TEDx Talks', 'TED-Ed','TEDxYouth','TED' ,'Use Demo Dataset'))
st.write('<p style="font-size:130%">Importing Real-time data through Youtube.</p>', unsafe_allow_html=True)

if file_data == 'TEDx Talks':
	playlist_id = channel_data.loc[channel_data['Channel_name']=='TEDx Talks', 'playlist_id'].iloc[0]
elif file_data == 'TED-Ed':
	playlist_id = channel_data.loc[channel_data['Channel_name']=='TED-Ed', 'playlist_id'].iloc[0]
elif file_data == 'TEDxYouth:
	playlist_id = channel_data.loc[channel_data['Channel_name']=='TEDxYouth', 'playlist_id'].iloc[0]
elif file_data == 'TED':
	playlist_id = channel_data.loc[channel_data['Channel_name']=='TED', 'playlist_id'].iloc[0]
elif file_data == 'Use Demo Dataset':
	data = 'TED_TALKS_DATA.csv'
#data = 'TED_TALKS_DATA.csv'

def get_video_ids(youtube, playlist_id):
    
    request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId = playlist_id,
                maxResults = 50)
    response = request.execute()
    
    video_ids = []
    
    for i in range(len(response['items'])):
        video_ids.append(response['items'][i]['contentDetails']['videoId'])
        
    next_page_token = response.get('nextPageToken')
    more_pages = True
    
    while more_pages:
        if next_page_token is None:
            more_pages = False
        else:
            request = youtube.playlistItems().list(
                        part='contentDetails',
                        playlistId = playlist_id,
                        maxResults = 50,
                        pageToken = next_page_token)
            response = request.execute()
    
            for i in range(len(response['items'])):
                video_ids.append(response['items'][i]['contentDetails']['videoId'])
            
            next_page_token = response.get('nextPageToken')
        
    return video_ids

video_ids = get_video_ids(youtube, playlist_id)

def get_video_details(youtube, video_ids):
    all_video_stats = []
    
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    id=','.join(video_ids[i:i+50]))
        response = request.execute()
        
        for video in response['items']:
            video_stats = dict(
                               Title = video['snippet']['title'],
                               Published_date = video['snippet']['publishedAt'],
                               Description=video['snippet']['description'],
                               Thumbnails = video['snippet']['thumbnails']['default']['url'],
                               Views = video['statistics']['viewCount'],
                               #Likes = video['statistics']['likeCount']
                               #URL = video['contentDetails']
                               #Dislikes = video['statistics']['dislikeCount'],
                               #Comments = video['statistics']['commentCount']
                               )
            all_video_stats.append(video_stats)
    
    return all_video_stats
video_details = get_video_details(youtube, video_ids)
video_data = pd.DataFrame(video_details)

video_data['Published_date'] = pd.to_datetime(video_data['Published_date']).dt.date
video_data['Views'] = pd.to_numeric(video_data['Views'])
video_data['Likes'] = pd.to_numeric(video_data['Likes'])
#video_data['Dislikes'] = pd.to_numeric(video_data['Dislikes'])
#video_data['Views'] = pd.to_numeric(video_data['Views'])
#video_data
data= video_data.to_csv('TED_TALKS_DATA.csv')

df = pd.read_csv(data)
data=df
st.subheader('Dataframe:')
n, m = df.shape
st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)   
st.dataframe(df)

#data = pd.DataFrame(data)
#st.write(data.head())
st.subheader("Language Detection")
from langdetect import detect
def det(x):
    try:
        language = detect(x)
    except:
        language = 'Other'
    return language
df['language'] = df['Description'].apply(det)
st.write(df)
st.subheader("Filtering English language")
filtered_for_english = df.loc[df['language'] == 'en']
df = df[df['language'] == 'en']
st.write(df)

df['details'] = df["Title"] + ' ' + df['Description']
 
st.subheader("Removing the unnecessary information")

df.dropna(inplace = True)
st.write(df)

st.subheader("Removing stopwords")
def remove_stopwords(text):
  stop_words = stopwords.words('english')
 
  imp_words = []
 
  # Storing the important words
  for word in str(text).split():
    word = word.lower()
     
    if word not in stop_words:
      imp_words.append(word)
 
  output = " ".join(imp_words)
 
  return output
df['details'] = df['details'].apply(lambda text: remove_stopwords(text))
st.write(df)

punctuations_list = string.punctuation


def cleaning_punctuations(text):
	signal = str.maketrans('', '', punctuations_list)
	return text.translate(signal)

st.subheader("Cleaning punctuations")
df['details'] = df['details'].apply(lambda x: cleaning_punctuations(x))
st.write(df)

details_corpus = " ".join(df['details'])

st.text("Training Model")
vectorizer = TfidfVectorizer(analyzer = 'word')
vectorizer.fit(df['details'])

def get_similarities(talk_content, data=df):

	# Getting vector for the input talk_content.
	talk_array1 = vectorizer.transform(talk_content).toarray()

	# We will store similarity for each row of the dataset.
	sim = []
	#pea = []
	for idx, row in data.iterrows():
		details = row['details']

		# Getting vector for current talk.
		talk_array2 = vectorizer.transform(
			df[df['details'] == details]['details']).toarray()

		# Calculating cosine similarities
		cos_sim = cosine_similarity(talk_array1, talk_array2)[0][0]

		# Calculating pearson correlation
		#pea_sim = pearsonr(talk_array1.squeeze(), talk_array2.squeeze())[0]

		sim.append(cos_sim)
		#pea.append(pea_sim)

	return sim #, pea

def recommend_talks(talk_content,n, data=df):
 
    df['cos_sim'] = get_similarities(talk_content)
 
    df.sort_values(by='cos_sim', ascending=
                     False, inplace=True)
 
    
    recommended_data = df.head(n)
    recommended_data.sort_values(by=['Views'],ascending=False)
    r_pic = recommended_data[["Thumbnails"]]
    r_name = recommended_data[["Title"]]
    st.subheader("Ted Talks you might like :- ")
    for i in range(n):
      pic =r_pic.iloc[i]["Thumbnails"]
      name = r_name.iloc[i]["Title"]
      #st.write("check out this [link](%s)" % url)
      
      st.write("Recommendation :- %s" %name)
      # URL of the image
      url= str(pic)
      image_url = url 
      # Fetch the image from the URL
      response = requests.get(image_url)
      image = Image.open(BytesIO(response.content))

      # Display the image in Streamlit
      #st.image(image, caption='Image Caption', use_column_width=True)
      desired_size = (240, 180)
      # Resize the image
      resized_image = image.resize(desired_size)
	
      # Display the resized image
      st.image(resized_image, caption='TED Talk Thumbnail')

	
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.subheader("Search for your TED talk here")
talk_content = [st.text_input(' Enter your Ted Talk keywords : ', "Life")]
n = st.number_input(' Enter number of recommendations you want ', 1)
#talk_content = [str(input(' Enter your Ted Talk keywords : '))]
recommend_talks(talk_content , n)

