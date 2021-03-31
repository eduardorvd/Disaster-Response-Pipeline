import sys
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import pickle
import pandas as pd
import re
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


def load_data(database_filepath):
    """
    Load Data from the SQLite database
    
    Input:
        database_filepath 
    Output:
        X -> a dataframe containing the messages
        Y -> a dataframe containing the labels
        category_names -> List of categories names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table ('DisasterTable2',engine)
    df = df[df.related != '2']
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.values
    return X, Y, category_names


def tokenize(text):
    """tokenize and transform input text. Return cleaned text"""
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Build the model with pipeline and Classifier"""
    pipeline_v2 = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
            
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    # Improved parameters 
    parameters = {
        'clf__estimator__n_estimators': [100, 200],
        #'clf__estimator__learning_rate': [0.1, 0.3]
     }
    # new model with improved parameters
    cv = GridSearchCV(estimator=pipeline_v2, param_grid=parameters, cv=3)
    #pipeline.get_params().keys()
    #cv.fit(X_train, y_train)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.values.astype(float), Y_pred.astype(float), target_names=category_names))
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])


def save_model(model, model_filepath):
    """Save model as pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
