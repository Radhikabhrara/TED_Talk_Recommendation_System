import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

from PIL import Image
# Loading Image using PIL
im = Image.open('ilogo.png')
# Adding Image to web app
st.set_page_config(layout = "wide",page_title="Radhika_1917631", page_icon = im)
st.title("TED Talks Recommendation System")
import streamlit.components.v1 as components

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
	
load_css('style.css')
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

def get_video_details(youtube, video_ids):
    all_video_stats = []
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
                    part='id,snippet,statistics,contentDetails',
                    id=','.join(video_ids[i:i+50]))
        response = request.execute()
        for video in response['items']:
            video_stats = dict(Id = video['id'],
                               Title = video['snippet']['title'],
                               Published_date = video['snippet']['publishedAt'],
                               Description=video['snippet']['description'],
                               Thumbnails = video['snippet']['thumbnails']['high']['url'],
                               Views = video['statistics']['viewCount']
                               )
            all_video_stats.append(video_stats)
    return all_video_stats

from googleapiclient.discovery import build
import pandas as pd

api_key= "AIzaSyDFgYvIgvBxnXSI6yenVx92ZfU_ud8go58"
channel_ids = ["UCsT0YIqwnpJCM-mx7-gSA4Q",
               "UCAuUUnT6oDeKwE6v1NGQxug",
               "UCsooa4yRKGN_zEE8iknghZA",
               "UC-yTB2bUcin9mmah36sXiYA",
	       "UCQSrdt0-Iu8qVEiJyzhrfdQ",
	       "UCDAdYdnCDt9zx3p3e_78lEQ"]        

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
channel_data['Views'] = pd.to_numeric(channel_data['Views'])
channel_data['Subscribers'] = pd.to_numeric(channel_data['Subscribers'])
channel_data['Total_videos'] = pd.to_numeric(channel_data['Total_videos'])
st.subheader("TED Talks Channel Data:")
st.write(channel_data)
fig = px.bar(channel_data, x='Channel_name', y='Subscribers', color="Channel_name" ,hover_name="Total_videos")
fig.update_layout(title='Subscribers Distribution among TED TAlk Channels:')
tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    # Use the Streamlit theme.
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    # Use the native Plotly theme.
    st.plotly_chart(fig, theme=None, use_container_width=True)
	
fig = px.bar(channel_data, x='Channel_name', y='Views', color="Channel_name" ,hover_name="Total_videos",template="plotly_dark")
fig.update_layout(title='Views on the videos among TED TAlk Channels:')
tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    st.plotly_chart(fig, theme=None,template="plotly_dark", use_container_width=True)

st.write('<p style="font-size:130%">Select TED talk Channel</p>', unsafe_allow_html=True)
file_data = st.radio('Channels List:', ('Use Demo Dataset','TEDx Talks', 'TED-Ed','TEDxYouth','TED' ,'TED Ideas Studio','TED Archive'))
st.write('<p style="font-size:130%">Importing Real-time data through Youtube.</p>', unsafe_allow_html=True)

if file_data == 'Use Demo Dataset':
	data = 'TED_TALKS_DATA.csv' 
else :
	playlist_id = channel_data.loc[channel_data['Channel_name']==file_data, 'playlist_id'].iloc[0]
	video_ids = get_video_ids(youtube, playlist_id)
	video_details = get_video_details(youtube, video_ids)
	video_data = pd.DataFrame(video_details)
	video_data['Published_date'] = pd.to_datetime(video_data['Published_date']).dt.date
	video_data['Views'] = pd.to_numeric(video_data['Views'])
	video_data.to_csv('TED_DATA.csv')
	data= 'TED_DATA.csv'
	
df = pd.read_csv(data)
data=df
st.subheader('Dataframe:')
n, m = df.shape
st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)   
st.dataframe(df)
st.text("Processing data....")
from langdetect import detect
def det(x):
    try:
        language = detect(x)
    except:
        language = 'Other'
    return language
df['language'] = df['Description'].apply(det)
df1=df

filtered_for_english = df.loc[df['language'] == 'en']
df = df[df['language'] == 'en']
df2=df

df['details'] = df["Title"] + ' ' + df['Description']
df.dropna(inplace = True)
df3=df

st.text("Few seconds away....")
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
df4=df
punctuations_list = string.punctuation
def cleaning_punctuations(text):
	signal = str.maketrans('', '', punctuations_list)
	return text.translate(signal)

#t.subheader("Cleaning punctuations")
df['details'] = df['details'].apply(lambda x: cleaning_punctuations(x))
df5=df

details_corpus = " ".join(df['details'])

st.sidebar.title("MENU:")
st.sidebar.header('Steps involved in Processing the data : 👉')
all_vizuals = ["Language Detection" ,"Filtering English language","Adding details & Removing the unnecessary information",
	      "Removing stopwords","Cleaning punctuations"]
#sidebar_space(3)         
vizuals = st.sidebar.multiselect("Choose which functionalities in processs you want to see 👇", all_vizuals)
if "Language Detection" in vizuals:
	st.subheader("Language Detection")
	st.write(df1)
if "Filtering English language" in vizuals:
	st.subheader("Filtering English language")
	st.write(df2)
if "Adding details & Removing the unnecessary information" in vizuals:
	st.subheader("Adding details & Removing the unnecessary information")
	st.write(df3)
if "Removing stopwords" in vizuals:
	st.subheader("Removing stopwords")
	st.write(df4)
if "Cleaning punctuations" in vizuals:
	st.subheader("Cleaning punctuations")
	st.write(df5)

st.text("Training Model.....")
vectorizer = TfidfVectorizer(analyzer = 'word')
vectorizer.fit(df['details'])

def get_similarities(talk_content, data=df):
	# Getting vector for the input talk_content.
	talk_array1 = vectorizer.transform(talk_content).toarray()
	# We will store similarity for each row of the dataset.
	sim = []
	for idx, row in data.iterrows():
		details = row['details']
		# Getting vector for current talk.
		talk_array2 = vectorizer.transform(
			df[df['details'] == details]['details']).toarray()
		# Calculating cosine similarities
		cos_sim = cosine_similarity(talk_array1, talk_array2)[0][0]
		sim.append(cos_sim)
	return sim 

def recommend_talks(talk_content,n, data=df):
	df['cos_sim'] = get_similarities(talk_content)
	df.sort_values(by='cos_sim', ascending= False, inplace=True)
	recommended_data = df.head(n)
	recommended_data['Views'] = pd.to_numeric(recommended_data['Views'])
	recommended_data.sort_values(by=['Views'],ascending=False)
	r_pic = recommended_data[["Thumbnails"]]
	r_name = recommended_data[["Title"]]
	r_view = recommended_data[['Views']]
	r_id =recommended_data[['Id']]
	st.subheader("Ted Talks you might like :- ")
	for i in reversed(range(n)):
		id_u = r_id.iloc[i]['Id']
		pic =r_pic.iloc[i]["Thumbnails"]
		name = r_name.iloc[i]["Title"]
		view =r_view.iloc[i]["Views"]
		id_ur = str(id_u)
		id_url = "http://www.youtube.com/watch?v=%s" %id_ur
		url= str(pic)
		image_url = url
		cap = "Views  :- %s" %view
		desired_size = (360, 270)
		st.write("%s" %name)
		response = requests.get(image_url)
		image = Image.open(BytesIO(response.content))
		st.image(image_url,caption=cap).resize(desired_size)
		# Generate the markdown code with the embedded URL
		markdown_code = f"[![image]({image_url})]({id_url})"
		# Render the markdown
		st.markdown(markdown_code, unsafe_allow_html=True)

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.sidebar.header('\n Switch to Recommendation system :  👇')
agree = st.sidebar.checkbox('I agree')
if agree:
    st.subheader("Search for your TED talk here")
    talk_content = [st.text_input(' Enter your Ted Talk keywords : ', "Life")]
    n = st.number_input(' Enter number of recommendations you want ', 6)
    recommend_talks(talk_content , n)
    
