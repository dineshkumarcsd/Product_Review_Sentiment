import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("amazon.csv")

# keep required columns
data = data[['review_content','rating']]

# convert rating to numeric
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')

# remove missing
data = data.dropna()

# create sentiment
data['sentiment'] = data['rating'].apply(lambda x: 1 if x >= 4 else 0)

# balance dataset
positive = data[data['sentiment']==1]
negative = data[data['sentiment']==0]

negative = negative.sample(len(positive), replace=True)

data = pd.concat([positive, negative])

X = data['review_content']
y = data['sentiment']

vectorizer = TfidfVectorizer(stop_words='english')

X_vec = vectorizer.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_vec,y,test_size=0.2)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train,y_train)

pickle.dump(model,open("model/sentiment_model.pkl","wb"))
pickle.dump(vectorizer,open("model/vectorizer.pkl","wb"))

print("Model trained successfully")