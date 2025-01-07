# spam-classification
# summary
Developed a robust machine learning pipeline to classify SMS messages as spam or ham using the SMS Spam Collection dataset. The dataset comprises 5,574 SMS messages, each tagged as either spam (unsolicited/irrelevant messages) or ham (legitimate messages). The objective was to build an efficient classification model to detect spam messages accurately.
Workflow Details:
1.	Text Preprocessing:
o	Lemmatization: Standardized words to their base forms (e.g., "running" → "run") to reduce dimensionality while preserving semantics.
o	Tokenization: Tokenized sentences using simple_preprocess from Gensim to break down SMS messages into words while removing punctuations and converting to lowercase.

2.	Feature Engineering:
o	Word2Vec Embeddings: Leveraged pre-trained Google News Word2Vec embeddings (300 dimensions) to represent each word as a semantic vector.
o	Aggregated Word Vectors (AvgWord2Vec): Computed the average word vector for each message to represent the entire SMS as a single feature vector.

3.	Model Development:
o	Random Forest Classifier: Trained a Random Forest Classifier using the message vectors as features. This model was chosen for its robustness and ability to handle feature importance effectively.

4.	Model Evaluation:
o	Conducted evaluation on a held-out test set of 1,112 SMS messages.
o	Precision, Recall, F1-Score, and Support:
	Ham (False): Precision = 0.92, Recall = 0.83, F1-Score = 0.88
	Spam (True): Precision = 0.98, Recall = 0.99, F1-Score = 0.98
o	Overall Metrics:
	Accuracy: 97%
	Weighted Avg F1-Score: 0.97
# Impact:
•	Built a highly accurate and reliable spam detection model that effectively distinguishes between legitimate and spam SMS messages.
•	The use of Word2Vec embeddings ensured semantic understanding of words, significantly enhancing model performance compared to traditional bag-of-words or TF-IDF approaches.
•	The pipeline can be adapted to other text classification tasks with minimal modifications, showcasing its versatility.
This project demonstrates expertise in text preprocessing, feature engineering with pre-trained embeddings, and classification with ensemble methods, making it a strong addition to a data scientist portfolio.
