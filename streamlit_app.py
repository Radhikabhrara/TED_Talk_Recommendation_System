import streamlit as st
import pandas as pd
import numpy as np
import sys

import tensorrt
tensorrt.nptype(tensorrt.DataType.HALF)
import io
import PIL
from PIL import Image
import dask.dataframe as dd

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

st.set_page_config(layout = "wide", page_title='Radhika_1917631')

st.title("TED Talks Recommendation System")

st.write('<p style="font-size:130%">Import Dataset</p>', unsafe_allow_html=True)
data = 'TED_TALKS_DATA.csv'
df = pd.read_csv(data)
st.subheader('Dataframe:')
n, m = df.shape
st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)   
   
#data= df = dd.read_csv('TED_TALKS_DATA.csv')
st.write(df)

#data = pd.DataFrame(data)
if 'numpy' in sys.modules:
    del sys.modules['numpy']

st.write(data.head())
#data = data.astype(bool)
st.table(data.head())
#data= np.bool_(False)

# Display the DataFrame using Streamlit
st.write(data.head())


st.subheader("Language Detection")
from langdetect import detect
def det(x):
    try:
        language = detect(x)
    except:
        language = 'Other'
    return language
df['language'] = df['Description'].apply(det)
st.write(data)
st.subheader("Filtering English language")
filtered_for_english = df.loc[df['language'] == 'en']
df = df[df['language'] == 'en']
st.write(data)

data['details'] = data["Title"] + ' ' + data['Description']
 
st.subheader("Removing the unnecessary information")
#data = data[['main_speaker', 'details',"name","url","title","views"]]
data.dropna(inplace = True)
st.write(data.head())

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
data['details'] = data['details'].apply(lambda text: remove_stopwords(text))
st.write(data.head())

punctuations_list = string.punctuation


def cleaning_punctuations(text):
	signal = str.maketrans('', '', punctuations_list)
	return text.translate(signal)

st.subheader("Cleaning punctuations")
data['details'] = data['details'].apply(lambda x: cleaning_punctuations(x))
st.write(data.head())

details_corpus = " ".join(data['details'])

st.text("Training Model")
vectorizer = TfidfVectorizer(analyzer = 'word')
vectorizer.fit(data['details'])

def get_similarities(talk_content, data=data):

	# Getting vector for the input talk_content.
	talk_array1 = vectorizer.transform(talk_content).toarray()

	# We will store similarity for each row of the dataset.
	sim = []
	#pea = []
	for idx, row in data.iterrows():
		details = row['details']

		# Getting vector for current talk.
		talk_array2 = vectorizer.transform(
			data[data['details'] == details]['details']).toarray()

		# Calculating cosine similarities
		cos_sim = cosine_similarity(talk_array1, talk_array2)[0][0]

		# Calculating pearson correlation
		#pea_sim = pearsonr(talk_array1.squeeze(), talk_array2.squeeze())[0]

		sim.append(cos_sim)
		#pea.append(pea_sim)

	return sim #, pea

def recommend_talks(talk_content,n, data=data):
 
    data['cos_sim'] = get_similarities(talk_content)
 
    data.sort_values(by='cos_sim', ascending=
                     False, inplace=True)
 
    
    recommended_data = data.head(n)
    recommended_data.sort_values(by=['Views'],ascending=False)
    r_pic = recommended_data[["Thumbnails"]]
    r_name = recommended_data[["Title"]]
    st.subheader("Ted Talks you might like :- ")
    for i in range(n):
      pic =r_pic.iloc[i]["Thumbnails"]
      name = r_name.iloc[i]["Title"]
      #st.write("check out this [link](%s)" % url)
      
      st.write("Recommendation :- %s" %name)
      #image = Image.open(pic)
      #response = requests.get(pic)
      #img = Image.open(BytesIO(response.content))

      #st.image(img, caption='Sunrise by the mountains')
      
      

st.subheader("Search for your TED talk here")
talk_content = [st.text_input(' Enter your Ted Talk keywords : ', "Life")]
n = st.number_input(' Enter number of recommendations you want ', 1)
#talk_content = [str(input(' Enter your Ted Talk keywords : '))]
recommend_talks(talk_content , n)

