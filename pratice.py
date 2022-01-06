import json
from eunjeon import Mecab
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
tokenizer = Mecab()

json_file = "./newsdata/NIRW2000000001.json"
with open(json_file, encoding="utf8") as j:
    content = json.loads(j.read())

life = []
gove = []
econ = []
texttoken=[]
newtext = []
topics = []
tokenizerr = Tokenizer(100)

''''
for i in tqdm(range(1000)):
    topic = content['document'][i]['metadata']['topic']
    if(topic == '생활'):
        for j in range(len(content['document'][i]['paragraph'])):
            token = tokenizer.morphs(content['document'][i]['paragraph'][j]['form'])
            clean = [word for word in token if not word in stopwords]
            life.append(clean)

for i in tqdm(range(1000)):
    topic = content['document'][i]['metadata']['topic']
    if(topic == '경제'):
        top
        for j in range(len(content['document'][i]['paragraph'])):
            token = tokenizer.morphs(content['document'][i]['paragraph'][j]['form'])
            clean = [word for word in token if not word in stopwords]
            econ.append(clean)

for i in tqdm(range(1000)):
    topic = content['document'][i]['metadata']['topic']
    if(topic == '정치'):
        for j in range(len(content['document'][i]['paragraph'])):
            token = tokenizer.morphs(content['document'][i]['paragraph'][j]['form'])
            #clean = [word for word in token if not word in stopwords]
            gove.append(token)
'''

for i in tqdm(range(100)):
    for j in range(len(content['document'][i]['paragraph'])):
        token = tokenizer.morphs(content['document'][i]['paragraph'][j]['form'])
        texttoken.append(token)

    newtext.append([element for array in texttoken for element in array])


for i in tqdm(range(100)):
    token = tokenizer.morphs(content['document'][i]['metadata']['topic'])
    topics.append(token)

texttoken = []


tokenizerr.fit_on_texts(topics)
topic_train = tokenizerr.texts_to_sequences(topics)
tokenizerr.fit_on_texts(newtext)
textdata_train = tokenizerr.texts_to_sequences(newtext)
textdata_train = pad_sequences(textdata_train, maxlen=170)
topic_train = pad_sequences(topic_train, maxlen=1)
#topic_train=to_categorical(topic_train)


embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(100,170))
model.add(LSTM(128))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
history = model.fit(textdata_train, topic_train, epochs=9,  batch_size=1)


