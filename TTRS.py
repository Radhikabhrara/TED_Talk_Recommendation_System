import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
nltk.download('stopwords')
warnings.filterwarnings('ignore')

from PIL import Image
# Loading Image using PIL
im = Image.open('logo.png')

# Adding Image to web app
st.set_page_config(page_title="Radhika_1917631", page_icon = im ,layout="wide",initial_sidebar_state="auto",
		   menu_items={'About': 'https://www.linkedin.com/in/radhika-bhrara/'})
st.title("TED Talks Recommendation System")

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

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

from langdetect import detect
def det(x):
	try:
		language = detect(x)
	except:
		language = 'Other'
	return language

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

punctuations_list = string.punctuation
def cleaning_punctuations(text):
	signal = str.maketrans('', '', punctuations_list)
	return text.translate(signal)

def get_similarities(talk_content, data):
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

def recommend_talks(talk_content,n, data):
	df['cos_sim'] = get_similarities(talk_content,data)
	df.sort_values(by='cos_sim', ascending= False, inplace=True)
	recommended_data = df.head(n)
	recommended_data['Views'] = pd.to_numeric(recommended_data['Views'])
	recommended_data.sort_values(by=['Views'],ascending=False)
	r_pic = recommended_data[["Thumbnails"]]
	r_name = recommended_data[["Title"]]
	r_view = recommended_data[['Views']]
	r_id =recommended_data[['Id']]
	st.subheader("Ted Talks you might like :- ")
	for i in range(n):
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

		markdown_code = f"[![image]({image_url})]({id_url})"
		# Render the markdown
		st.markdown(markdown_code, unsafe_allow_html=True)
		st.caption(cap)
		#st.text(cap)
		st.write('\n \n \n ')

st.sidebar.title("Menu Bar:")
rad=st.sidebar.radio("Navigation👉",["Home","Selecting the dataset :"])
if rad=="Home":
	image = 'TED.gif'
	st.image(image, caption='TED TALKS ')
	st.header('\n\n\nProject submission ')
	st.subheader('Radhika --1917631')

if rad=="Selecting the dataset :":
	channel_statistics = get_channel_stats(youtube, channel_ids)
	channel_data = pd.DataFrame(channel_statistics)
	channel_data['Views'] = pd.to_numeric(channel_data['Views'])
	channel_data['Subscribers'] = pd.to_numeric(channel_data['Subscribers'])
	channel_data['Total_videos'] = pd.to_numeric(channel_data['Total_videos'])
	st.subheader("TED Talks Channel Data:")
	st.write(channel_data)
	st.sidebar.header('Check distribution in youtube Channels data : 👉')
	all_vizuals = ["Subscribers Distribution" ,"Views Distribution"]
	       
	vizuals = st.sidebar.multiselect("Choose visualizations 👇", all_vizuals)
	if "Subscribers Distribution" in vizuals:
		fig = px.bar(channel_data, x='Channel_name',  y='Subscribers', color="Channel_name" ,hover_name="Total_videos",template="plotly_dark")
		fig.update_layout(title='Subscribers on the videos among TED TAlk Channels:')
		tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
		with tab1:
			# Use the Streamlit theme.
			st.plotly_chart(fig, theme="streamlit", use_container_width=True)

		with tab2:
			# Use the native Plotly theme.
			st.plotly_chart(fig, theme=None, use_container_width=True)
	if "Views Distribution" in vizuals:
		fig = px.bar(channel_data, x='Channel_name', y='Views', color="Channel_name" ,hover_name="Total_videos")
		fig.update_layout(title='Views Distribution among TED TAlk Channels:')
		tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
		with tab1:
			# Use the Streamlit theme.
			st.plotly_chart(fig, theme="streamlit", use_container_width=True)

		with tab2:
			# Use the native Plotly theme.
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
	st.subheader('\n Dataframe:')
	n, m = df.shape
	st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)   
	st.dataframe(df)
		
	st.header("Pre-processing data....")
	st.sidebar.header("Pre-processing the dataset :👉")

	df['language'] = df['Description'].apply(det)
	df1=df
	st.subheader("Language Detection:")
	st.write(df1)
	
	st.subheader("Filtering English Language:")
	filtered_for_english = df.loc[df['language'] == 'en']
	df=df[df['language'] == 'en']
	df2=df
	st.write(df2)
	
	st.subheader("Adding Details attribute for natural language processsing: ")
	df['details'] = df["Title"] + ' ' + df['Description']
	df.dropna(inplace = True)
	df3=df
	st.write(df3)
	
	df['details'] = df['details'].apply(lambda text: remove_stopwords(text))
	st.text("\n Few seconds away....")
	st.subheader("Removing stopwords :")
	df4=df["details"]
	st.write(df4)
	
	df['details'] = df['details'].apply(lambda x: cleaning_punctuations(x))
	st.subheader("Cleaning details by removing puctuations :")
	df5=df["details"]
	st.write(df5)
	
	details_corpus = " ".join(df['details'])
	genre = st.radio("Visual Representation of Content in TED Channel",
			 ("WordCloud"))
	if genre == 'WordCloud':
		st.subheader('Word Cloud of the TED Talk details')
		plt.figure(figsize=(20, 20))
		wc = WordCloud(max_words=1000, width=800, height=400).generate(details_corpus)
		plt.axis('off')
		st.plt(wc)

	st.text("Training Model.....")
	vectorizer = TfidfVectorizer(analyzer = 'word')
	vectorizer.fit(df['details'])
	
	st.sidebar.header('\n Switch to Recommendation system :  👇')
	agree = st.sidebar.checkbox('I agree')
	if agree:
		st.header("Recommendation System:")
		st.subheader("Search for your TED talk here")
		talk_content = [st.text_input(' Enter your Ted Talk keywords : ', "Life")]
		n = st.number_input(' Enter number of recommendations you want ', 6)
		recommend_talks(talk_content , n ,df)
	
