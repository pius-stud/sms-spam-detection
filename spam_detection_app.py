## Creating a Web Interface with Streamlit

# Through this code, it will be possible to create a web interface, running on a local port, 
# where we can use either a recently trained classifier or a
# Large Language Model (LLM) - specifically Flan-T5-XXL - for our classification task.


# Import for LLM
import os
from langchain import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate

# Import text preprocessing
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Import for the classifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Import for data
from conversion import arff_to_csv
import pandas as pd

# Import for the interface
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PIL import Image


# LLM
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<INSERT_HERE_YOUR_TOKEN>"


# Load the dataset
df = arff_to_csv(
    url="https://storm.cis.fordham.edu/~gweiss/data-mining/weka-data/sms-spam-dataset.arff",
    columns=['Text', 'Class'],
    name_csv='spam_det.csv')


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


# Train Classifier
from sklearn.model_selection import train_test_split

# Split data into features (X) and target (y)
X = df['Text']
y = df['Class']

# Extract features from text data using TF-IDF
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the classifier
classifier = MultinomialNB()
classifier.fit(X_train.toarray(), y_train)



# App

# Sidebar contents
with st.sidebar:
    add_vertical_space(1)
    st.title("UNISA SPAM DETECTOR")
    image = Image.open('./unisa.png')
    col1, col2, col3 = st.columns([0.82, 2, 0.82])
    col2.image(image, use_column_width=True)
    add_vertical_space(2)
    st.title('🤗💬 Method')
    source = st.radio("Select the method to use for verification", ('Trained Classifier', 'LLM - Flan-T5-XXL'))




# Main function
def main():
    st.header("SMS SPAM DETECTOR 💬")
    query = st.text_input("Insert the SMS to verify:")

    if query:
        # To use the classifier
        if source == 'Trained Classifier':
            sms_test_vectorized = vectorizer.transform([query])
            prediction = classifier.predict(sms_test_vectorized.toarray())
            
            if prediction[0] == 'ham':
                image = Image.open('./success.png')
            else:
                image = Image.open('./reject.png')

            st.write(prediction[0])
            st.image(image, width=75)

        # To use an LLM
        elif source == 'LLM - Flan-T5-XXL':
            FEW_SHOT = '''Your task is to classify a message as 'spam' - if this is a spam message - or 'ham' - if not.
            
Message: 'Go until jurong point crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..'
Answer: ham

Message: 'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&Cs apply 08452810075over18s'
Answer: spam

Message: 'Ok lar... Joking wif u oni...'
Answer: ham

Message: 'FreeMsg Hey there darling its been 3 weeks now and no word back! Id like some fun you up for it still? Tb ok! XxX std chgs to send å£1.50 to rcv'
Answer: spam

Message: 'As per your request Melle Melle (Oru Minnaminunginte Nurungu Vettam) has been set as your callertune for all Callers. Press *9 to copy your friends Callertune'
Answer: ham

Message: 'WINNER!! As a valued network customer you have been selected to receivea å£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.'
Answer: spam

Message: 'Even my brother is not like to speak with me. They treat me like aids patent.'
Answer: ham

Message: 'Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030'
Answer: spam

{question}'''

            repo_id = "google/flan-t5-xxl"
            llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 1024})
            template = FEW_SHOT
            prompt = PromptTemplate(template=template, input_variables=["question"])
            llm_chain = LLMChain(prompt=prompt, llm=llm)

            question = f"""Message:{query}
Answer:"""

            response = llm_chain.run(question)

            if response == 'ham':
                image = Image.open('./success.png')
            else:
                image = Image.open('./reject.png')

            st.image(image, width=75)
            st.write(response)


if __name__ == '__main__':
    main()
