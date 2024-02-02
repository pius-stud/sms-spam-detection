# CREIAMO UN'INTERFACCIA WEB CON STREAMLIT

# Attraverso questo codice, sarà possibile ottenere un'interfaccian web, eseguita su una porta locale,
# nella quale potremo sfruttare per il nostro task di classificazione un classificatore appena addestrato 
# oppure un LLM (Large Langauge Model) - per la precisione Flan-T5-XXL.


# Import per LLM
import os
from langchain import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate


# Import text preprocessing
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Import per classificatore
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Import per i dati
from conversion import arff_to_csv
import pandas as pd


# Import per l'interfaccia
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PIL import Image



##############
# LLM
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<INSERT_HERE_YOUR_TOKEN>"



##############################
# Carichiamo il dataset
df = arff_to_csv(
    url = "https://storm.cis.fordham.edu/~gweiss/data-mining/weka-data/sms-spam-dataset.arff",
    columns = ['Text', 'Class'],
    name_csv = 'spam_det.csv')


##############################
# Text Preprocessing


nltk.download('all')

text = list(df['Text'])

lemmatizer = WordNetLemmatizer()

corpus = []
for i in range(len(text)):

    r = re.sub('[^a-zA-Z]', ' ', text[i])

    r = r.lower()

    r = r.split()

    r = [word for word in r if word not in stopwords.words('english')]

    r = [lemmatizer.lemmatize(word) for word in r]

    r = ' '.join(r)

    corpus.append(r)


df['Text'] = corpus


##############################
# ADDESTRAMENTPO CLASSIFICATORE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Dividiamo i dati in feature (X) e target (y)
X = df['Text']
y = df['Class']

# Estraiamo le feature dai dati testuali utilizzando TF-IDF
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Dividiamo i dati in set di addestramento e set di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creiamo e addestriamo il classificatore
classifier = MultinomialNB()
classifier.fit(X_train.toarray(), y_train)

##############################



# APP

# Sidebar contents
with st.sidebar:

    # PER INSERIRE L'IMMAGINE DI UNISA NELLA BARRA LATERALE Sx

    add_vertical_space(1)
    st.title("UNISA SPAM DETECTOR")
    image = Image.open('./unisa.png')
    col1, col2, col3 = st.columns([0.82, 2, 0.82])
    col2.image(image, use_column_width=True)

    # PER SCEGLIERE COME CLASSIFIACRE IL MESSAGGIO
    add_vertical_space(2)
    st.title('🤗💬 Method')

    # Mostriamo i metodi possibili
    source = st.radio("Select the method to use for verification",('Trained Classifier', 'LLM - Flan-T5-XXL'))


# # # # #


def main():

    st.header("SMS SPAM DETECTOR 💬")


    query = st.text_input("Insert the SMS to verify:")


    if query:

      # Per usare il classificatore
      if source == 'Trained Classifier':

        # Vettorizziamo la stringa (sms su cui fare la prediction)
        sms_test_vectorized = vectorizer.transform([query])


        # Efftuiamo la predizione sfruttando il classificatore
        # addestrato in precedenza
        prediction = classifier.predict(sms_test_vectorized.toarray())


        if prediction[0]=='ham':
          image = Image.open('./success.png')
        else:
          image = Image.open('./reject.png')

        st.write(prediction[0])
        st.image(image,width=75)


      # Per usare un LLM
      elif source == 'LLM - Flan-T5-XXL':


        # Creiamo un Prompt Few-Shot per far sì che il modello impari il task dagli esempi forniti

        FEW_SHOT = '''Your task is to classify a message as 'spam' - if this is a spam message - or 'ham' - if not.


Messagge: 'Go until jurong point crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..'
Answer: ham

Messagge: 'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&Cs apply 08452810075over18s'
Answer: spam

Messagge: 'Ok lar... Joking wif u oni...'
Answer: ham

Messagge: 'FreeMsg Hey there darling its been 3 weeks now and no word back! Id like some fun you up for it still? Tb ok! XxX std chgs to send å£1.50 to rcv'
Answer: spam

Messagge: 'As per your request Melle Melle (Oru Minnaminunginte Nurungu Vettam) has been set as your callertune for all Callers. Press *9 to copy your friends Callertune'
Answer: ham

Messagge: 'WINNER!! As a valued network customer you have been selected to receivea å£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.'
Answer: spam

Messagge: 'Even my brother is not like to speak with me. They treat me like aids patent.'
Answer: ham

Messagge: 'Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030'
Answer: spam

{question}'''

        # Definiamo la repo di HuggingFace del modello
        repo_id = "google/flan-t5-xxl"

        # Definiamo il modello ottenuto da Huggingface grazie a LangChain
        llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 1024})


        # Adesso definiamo qual è il template di prompt che il modello deve seguire per rispondere
        template = FEW_SHOT

        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt,llm=llm)


        # La nostra domanda è ciò che abbiamo inserito in 'query'
        # 'question' sarà inserita nel template

        question = f"""Message:{query}
Answer:"""


        # Interroghiamo il modello
        response = llm_chain.run(question)

        # In base alle risposta del modello, avremo un'immagine o un'altra
        if response=='ham':
            image = Image.open('./success.png')
        else:
            image = Image.open('./reject.png')


        # STAMPA DELL'IMMAGINE E DEI RELATIVI LINK
        st.image(image,width=75)
        st.write(response)



##############################
if __name__ == '__main__':
    main()
