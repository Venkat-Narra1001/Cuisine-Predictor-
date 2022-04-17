from unittest import result
import project2
import pandas as pd

def test_clean_data():
    text= ['rice', '1% milk', 'beens','All purpose Flour','Vegetable Oil']
    test=project2.clean_text(text)
    assert test=='rice milk beens all purpose flour vegetable oil'

def test_jaccardsimilarity():
    text1='rice wheat all purpose flour vegetable oil'
    text2='rice wheat all purpose flour vegetable oil'
    text3='rice wheat all purpose flour vegetable oil'
    text4='black pepper shallots salmon sugar'
    test=project2.jaccardSimilarity(text1,text2)
    test2=project2.jaccardSimilarity(text3,text4)
    assert test==1 and test2==0

