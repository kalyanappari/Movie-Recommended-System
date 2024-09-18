import pandas as pd
import numpy as np
import ast #used to convert the string of list to list.
import nltk 

movies = pd.read_csv(r"C:\Users\asaik\OneDrive\Documents\Movie Recomemded System\tmbd_movies.csv")
credits = pd.read_csv(r"C:\Users\asaik\OneDrive\Documents\Movie Recomemded System\tmdb_5000_credits.csv",low_memory=False)
#Print the head
#print(movies.head())
#print(credits.head())

"""print(credits.head(1)['cast'].values)
print(credits.head(1)['crew'].values)"""


#merging the data frames based on the coloum title.

"""print(movies.merge(credits,on='title').shape)

print(movies.shape)

print(credits.shape)"""

movies = movies.merge(credits,on='title')

#print(movies.head(1))

#feature extraction.

#print(movies.info())

'''
1.genres
2.id
3.keywords
4.title
5.overview
6.cast
7.crew

'''

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

#print(movies.head())

#data preprocessing.
#handling the missing values in the selected features.

#print(movies.isnull().sum())

movies.dropna(inplace=True) # missing values rows were dropped.

#print(movies.isnull().sum())

#chechking duplicancency.

#print(movies.duplicated().sum()) ## no duplicency found.

#print(movies.iloc[0].genres)

#[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]

#just extracting the important name from the wieared format of the genres.

#['Action','Adventures','fantacy','science fiction']

#to convert we use the convert function.

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)


#print(movies.head()['genres'])

#print(movies.head())

#print(movies.head(1)['cast'])

#extracting the frist three Actors names from the cast feature.

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)

#extracting the director name from the crew feature.

def fetch_director(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])

            
            break
    return L
movies['crew'] = movies['crew'].apply(fetch_director)

#print(movies.head(2))

print(movies['overview'][0]) #string.

#converting the string into one list.

movies['overview'] = movies['overview'].apply(lambda x:x.split())

#print(movies.head())

#removies the spaces between the two name of a same string due to confusion while combining some features into the one tag.

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


#print(movies.head())

#creating the tags coloumn by concatinating the overview ,genre , keywords , cast , crew.

movies['tags'] =  movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

#print(movies.head())

#crating the new data set by extracting some coloumns.

new_data = movies[['movie_id','title','tags']]

#print(new_data.head())

#converting the tag lists into the list.

new_data.loc[:, 'tags'] = new_data['tags'].apply(lambda x: " ".join(x))



print(new_data.head())



from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_data['tags'] = new_data['tags'].apply(stem)

#print(new_data['tags'][0])

'''In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron'''

#print(new_data['tags'][1])

'''Captain Barbossa, long believed to be dead, has come back to life and is headed to the edge of the Earth with Will Turner and Elizabeth Swann. But nothing is quite as it seems. Adventure Fantasy Action ocean drugabuse exoticisland eastindiatradingcompany loveofone'slife traitor shipwreck strongwoman ship alliance calypso afterlife fighter pirate swashbuckler aftercreditsstinger JohnnyDepp OrlandoBloom KeiraKnightley GoreVerbinski'''

#vextorization.
#converting the text data to the vector.


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vectors = cv.fit_transform(new_data['tags']).toarray()

#print(vectors)

#print(cv.get_feature_names())

#calculating the distance between the vectors.

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

print(similarity)

def recommend(movie):
    movie_index = new_data[new_data['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]
    print("Recommend List:")
    for i in movies_list:
        print(new_data.iloc[i[0]].title)

#movie = str(input("Enter the movie: "))

recommend("Avatar")

import pickle
pickle.dump(new_data,open('movies.pkl','wb'))

pickle.dump(new_data.to_dict(),open('movie_dict.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))

