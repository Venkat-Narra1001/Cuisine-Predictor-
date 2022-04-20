# cs5293sp22-project2

## Name: Venkat Narra

## How to run the project

``` bash
pipenv run python project2.py --N 5 --ingredient salt --ingredient wheat --ingredient "vegetable oil" --ingredient water
```
By running the above command in the compiler, it takes arguments  ``` --N``` to genrate ouput that contains top N matching recipies. ```--ingrediant``` eah ingredient of a recipe is givrn by this argument.The program will genarate a prediction of cuisine type of that ingredients and top n recipes with Id and similarity score.


## External libraries that are used in the project

1. **sckit-learn**
    ```bash
    pipenv install scikit-learn
    ```
   - It's used for accessing the Machine learning model Logestic regression.
   - This model shows the highst accuracy of prediction compared to other models.
### Bugs
1. If the entered ingredients are not ingredients or if they are not in the voacbulary genrated by the vertex its predicting randomly.

### Assumption

1. score below the predicted cuisine type is ,score that the ingredients avilable in that particular cuisine.

## Functions in this project

1. **clean_text()**
    - This function takes the list that contains all the ingredients given by the user and makes them as astring by cleaning the text.

2. **jaccardSimilarity()**
    - This function genrates a similarity score between the two strings and returnss score.

3. **topn()**
    - This function will genarte a list that contains a Id and similarity score of input ingredients and ingredientsof that particular cuisine.

# Tests

1. **test_clean_data()**
    - In this function ,I am checking weather the then function clearing all the required data from the ingredients list sent to this  function.

2. **test_jaccardsimilarity()**
    - In this test, I am checking weather the score genarateing as expected or not.  

3. **test_topn()**
    - In this test i am manually sendeing a dataframe data to the function and asserting the expected output,weather its returning the list that contain id and similarity score of the given N values. 