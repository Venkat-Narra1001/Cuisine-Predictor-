import json
from pickle import NONE
from tkinter import N
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')


def main(args):
    '''
    This main function takes the aruguments entered in the command line and perform prediction process.
    Parameters
    ----------
    args:
        arguments entered in the command line.

    Returns
    -------
            print the predictions.
    '''
    # loading the data from json file
    data = pd.read_json("yummly.json").set_index('id',drop = False)
    df = pd.DataFrame(data)
    df['ingredients']=df['ingredients'].map(lambda x: clean_text(x))


    ingrediants_data=list(df['ingredients'])
    # vectorizing the ingredients data
    vectorizer = TfidfVectorizer()
    x=vectorizer.fit_transform(ingrediants_data)
    '''
    Here tspliting the data for training and testing and loading the Logistic Regression model and
    loading the trained model in a pickle file.
    '''
    X_train, X_test, y_train, y_test = train_test_split( x.toarray(), df['cuisine'], test_size=0.10)
    # lr = LogisticRegression(max_iter=1000)
    # lr.fit(X_train, y_train)
    # pickle.dump(lr,open('logestic.sav','wb'))

    lr_m=pickle.load(open('logestic.sav','rb')) # using the saved model
    # Vectorizing the input data.
    input_data=[clean_text(args.ingredient)]
    vectorizer2 = TfidfVectorizer(vocabulary=vectorizer.vocabulary_)
    vector2=vectorizer2.fit_transform(input_data).toarray()
    feature_names=vectorizer2.get_feature_names()
    df_test=pd.DataFrame(vector2,columns=feature_names)
    #predicting the type of cuisine using saved model.
    cuisine_predicted=lr_m.predict(df_test)

    if args.N != NONE:
        # identyifying the topN recipies matches the recipies given. 
        n=topn(df,input_data)
        cuisines=df.groupby('cuisine')
        # identifying the score for tge predicted cusine
        predicted_cuisine_list=cuisines.get_group(cuisine_predicted[0])
        score_predicted=len([ x for x in list(predicted_cuisine_list['similarity_score']>0) if x==True])/predicted_cuisine_list['cuisine'].count()
    # grouping the topN recipies in a list.
    topn_values=[]
    for i,j in n:
        topn_values.append({'id':i,'score':"{:.2f}".format(j)})

    out={
        "cuisine": f"{cuisine_predicted[0]}",
        "score" : "{:.2f}".format(score_predicted),
        "closest": topn_values
    }
        
    print(json.dumps(out))
    
def clean_text(lst):
    '''
    This funtion clean the text data in a list.
    Parameters
    ----------
    lst:
        ingredients in a list.

    Returns
    -------
            cleaned string that contains ingredients.
    '''
    lst = [ x.lower() for x in lst]
    lst=" ".join(lst)
    lst=re.sub('[^A-Za-z\s]+',' ',lst)
    lst=re.sub(' +', ' ',lst)
    return lst

def jaccardSimilarity(Sentence1, Sentence2):
    '''
    This funtion check the similarity between two sentences.
    Parameters
    ----------
    sentences:
            two sentences that need the similarity check

    Returns
    -------
           score that matches the two strings.
    '''
    a = set(Sentence1.split())
    b = set(Sentence2.split())
    c = a.intersection(b)
    Similarity = float(len(c))/(len(a)+len(b)-len(c))
    return Similarity

def topn(df,input_data):
    '''
    This funtion clean the text data in a list.
    Parameters
    ----------
    lst:
        ingredients in a list.

    Returns
    -------
            cleaned string that contains ingredients.
    '''
    similarity_score_ingredients=[]
    for i in df['ingredients']:
        similarity_score_ingredients.append(jaccardSimilarity(input_data[0],i))
    df['similarity_score']=similarity_score_ingredients
    sorted_score=df.sort_values('similarity_score',ascending=False)[:args.N]
    id=list(sorted_score['id'])
    scores=list(sorted_score['similarity_score'])
    id_score=list(zip(id,scores))
    return id_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int,default= 2, help='<Required> Set flag',required=True )
    parser.add_argument('--ingredient', action='append', help='<Required> Set flag', required=True)
    args = parser.parse_args()
    main(args) 