import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import sklearn as sk
import random
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer


np.random.seed(42)
random.seed(42)
nltk.download('stopwords')
nltk.download('wordnet')

LOWERCASE = False
PUNCTATION_REMOVAL = True
STOPWORD_REMOVAL = False
LEMMATIZE = False
STEMMING = False

def preprocess_corpus(sentences: list[str]):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))
    processed = []
    for sen in sentences:
        lower = sen
        # lowercasing elements
        if LOWERCASE:
            lower = sen.lower()
        
        if len(lower) < 5:
            continue

        # removal of punctuation
        if PUNCTATION_REMOVAL:
            lower = ''.join(char for char in lower if char.isalnum() or char.isspace())

        # removal of stopwords
        words = lower.split()
        
        if STOPWORD_REMOVAL:
            words = [word for word in words if word not in stopwords_set]

        if LEMMATIZE:
            words = [lemmatizer.lemmatize(word) for word in words]

        if STEMMING:
            words = [stemmer.stem(word) for word in words]

        processed.append(' '.join(words))
    return processed


def extract_features_sentiment(corpus:list[(str, str)], vectorizer:TfidfTransformer=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            ngram_range=(1,2),
            min_df=2,
            lowercase=LOWERCASE
            )
        
        vec = vectorizer.fit_transform(corpus)
    else:
        vec = vectorizer.transform(corpus)
    

    neg_counts = []
    negative_words = ['no','not','n\'t','bad']
    for line in corpus:
        count = 0
        for word in negative_words:
            count += line.count(word)
        neg_counts.append(count)
    
    vec = np.array(vec.todense())
    neg_counts = np.array(neg_counts).reshape(-1,1)
    vec = np.hstack((neg_counts, vec))
    
    features = np.concat([vectorizer.get_feature_names_out(), ['NEGATIVE WORDS']])

    return vec, vectorizer, features

def extract_features_plural(corpus:list[(str, str)], vectorizer:TfidfTransformer=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            ngram_range=(1,3),
            min_df=2,
            lowercase=LOWERCASE,
            analyzer='char_wb'
            )
        
        vec = vectorizer.fit_transform(corpus)
    else:
        vec = vectorizer.transform(corpus)
    
    features = vectorizer.get_feature_names_out()
    return vec, vectorizer, features


def train_models(train_set, train_labels):
    """trains and returns a logistic regression model and a linear SVC"""
    
    parameters = {
        'penalty': ['l1','l2'],
        'C': np.logspace(-3,3,7)
        }
    logReg = LogisticRegression(solver='liblinear')
    
    logReg_search = GridSearchCV(
        estimator=logReg,
        param_grid=parameters,
        cv=5,
        scoring='accuracy', 
        verbose=0,
        n_jobs=-1
    )
    logReg_search.fit(train_set, train_labels)
    
    print("---------------------------")
    print("Logistic Regression Best Params:", logReg_search.best_params_)
    print("Best score: ", logReg_search.best_score_)


    svc = LinearSVC(max_iter=90000)
    svc_search = GridSearchCV(
        estimator = svc,
        param_grid=parameters,
        cv=5,
        scoring='accuracy', 
        verbose=0,
        n_jobs=-1
    )
    svc_search.fit(train_set, train_labels)

    print("---------------------------")
    print("Linear SVC Best Params:", svc_search.best_params_)
    print("Best score: ", svc_search.best_score_)
    print("---------------------------")


    return logReg_search.best_estimator_, svc_search.best_estimator_

def read_data(filename, classification):
    data = []
    val = []
    with open(filename, 'r') as f:
        data.extend(f.readlines())
    for _ in data:
        val.append(classification)
    return data, val

def evaluate_model(model, x_test, y_test):
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    conf_mat = confusion_matrix(y_test, preds)

    print(f'MODEL METRICS:\n \tACCURACY: {acc} \n\tPRECISION: {precision}\n\tRECALL: {recall}')
    return (acc, precision, recall, conf_mat)
    

def run_full_pipeline(filenames, feature_extractor, metricsFilename):
    corpus, vals = read_data(f'RA1/{filenames[0]}', 0)
    corpus1, vals1 = read_data(f'RA1/{filenames[1]}', 1)
    corpus.extend(corpus1)
    vals.extend(vals1)

    corpus = preprocess_corpus(corpus)
    
    X_train, X_test, Y_train, Y_test = train_test_split(corpus, vals, test_size=0.1)

    train_feats, vectorizer, features = feature_extractor(X_train)
    test_feats, _, _ = feature_extractor(X_test, vectorizer)

    logReg, svc = train_models(train_feats, Y_train)

    logReg_metrics = evaluate_model(logReg, test_feats, Y_test)
    svc_metrics = evaluate_model(svc, test_feats, Y_test)
    
    write_metrics(logReg_metrics, svc_metrics, metricsFilename)

    print("-----Logistic Regression Best Coeffs-----")
    find_strongest_coefficients(logReg, features)
    print("-----Linear SVC Best Coeffs-----")
    find_strongest_coefficients(svc, features)

def write_metrics(logMetrtics, svcMetrics, filename):
    with open(f"RA1/LogisticRegression_{filename}", 'a+') as f:
        if len(f.readlines()) == 0:
            f.write("Accuracy, Precision, Recall, Confusion Matrix\n")
        for el in logMetrtics:
            stringContent = str(el).replace('\n',' ')
            f.write(f"{stringContent},")
        f.write("\n")
    
    with open(f"RA1/LinearSVC_{filename}", 'a+') as f:
        if len(f.readlines()) == 0:
            f.write("Accuracy, Precision, Recall, Confusion Matrix\n")
        for el in svcMetrics:
            stringContent = str(el).replace('\n',' ')
            f.write(f"{stringContent},")
        f.write("\n")
    
def find_strongest_coefficients(model, features):
    coefs = model.coef_[0]
    indices = np.argsort(coefs)

    most_neg = indices[:10]
    most_pos = indices[-10:][::-1]
    
    print("Most Negative Features: ")
    for neg in most_neg:
        print(f"\t{features[neg]}: {coefs[neg]}")
    
    print("\nMost Positive Features: ")
    for pos in most_pos:
        print(f"\t{features[pos]}: {coefs[pos]}")

def clean_files(filename):  
    with open(filename, "r") as f:
        content = f.readlines()
    with open(filename, 'w') as f:
        for line in content:
            if len(line) > 5:
                f.write(line)
    
if __name__ == '__main__':
    files = ['synsem0.txt','synsem1.txt','morphphon0.txt','morphphon1.txt']
    sentiment_files = ['synsem0.txt','synsem1.txt']
    plural_files = ['morphphon0.txt','morphphon1.txt']
    for file in files:
        clean_files(f"RA1/{file}")
    run_full_pipeline(sentiment_files, extract_features_sentiment, "SentimentDetectionPerformance.csv")
    run_full_pipeline(plural_files, extract_features_plural, "PluralDetectionPerformance.csv")