# sms-spam-detection

### Spam_detector.ipynb

In order to create a classifier capable of detecting whether a given message is spam or not, we structured the notebook `Spam_detector.ipynb` by not training a single classifier with only one type of Text Preprocessing. Instead, we tested various approaches and classification algorithms to compare different solutions.

- First, we imported the data (section 1. Data Import) and conducted a descriptive analysis.

- Afterwards, we wanted to test lemmatization first and stemming later to perform Text Preprocessing (section 2. Text Preprocessing Lemmatization and section 3. Text Preprocessing - Stemming).

- In these two sections (section 2 and 3), we trained 5 different algorithms - Dummy Classifier, MultinomialNB, Support Vector Machine, Decision Tree, and Logistic Regression - using two different feature extraction strategies, namely Bag of Words (CountVectorizer) and TF-IDF (TfidfVectorizer), resulting in a total of 20 classifiers.

- In the penultimate section (section 4. Accuracy of the 20 classifiers), we decided to collect the accuracies of all trained classifiers to compare their performances.

- Finally, we implemented a Hard Voting Strategy (section 5. Handling Class Imbalance) by training 7 classifiers on datasets with balanced classes. Combining the various outcomes, the most frequently returned result will be the final one.
