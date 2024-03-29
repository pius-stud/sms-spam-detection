# sms-spam-detection

## conversion.py

`conversion.py` is a module where has been defined a function useful to open an url of a file in format `.arff` and give us back a `.csv` file.
This function has been used in the other two scripts: `Spam_detector.ipynb` and `spam_detection_app.py`.

## Spam_detector.ipynb

In order to create a classifier capable of detecting whether a given message is spam or not, we structured the notebook `Spam_detector.ipynb` by not training a single classifier with only one type of Text Preprocessing. Instead, we tested various approaches and classification algorithms to compare different solutions.

- First, we imported the data (section 1. Data Import) and conducted a descriptive analysis.

- Afterwards, we wanted to test lemmatization first and stemming later to perform Text Preprocessing (section 2. Text Preprocessing Lemmatization and section 3. Text Preprocessing - Stemming).

- In these two sections (section 2 and 3), we trained 5 different algorithms - Dummy Classifier, MultinomialNB, Support Vector Machine, Decision Tree, and Logistic Regression - using two different feature extraction strategies, namely Bag of Words (CountVectorizer) and TF-IDF (TfidfVectorizer), resulting in a total of 20 classifiers.

- In the penultimate section (section 4. Accuracy of the 20 classifiers), we decided to collect the accuracies of all trained classifiers to compare their performances.

- Finally, we implemented a Hard Voting Strategy (section 5. Handling Class Imbalance) by training 7 classifiers on datasets with balanced classes. Combining the various outcomes, the most frequently returned result will be the final one.

## spam_detection_app.py

The python code in the file `spam_detection_app.py` could be used in your Command Prompt (Note: I used the Anaconda terminal related to the environment of this script) to have a Web UI running on your local machine and interact with two spam classifiers: a Trained One and an LLM (Flan-T5-XXL). You need an API token 'read' from Hugging Face to employ the LLM as a classifier.


**(Note: Comments in code and Texts in markdown cells are in Italian, but code is universal 😏)**
