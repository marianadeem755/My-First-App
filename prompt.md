# Prompt:
1. Hey, CHAT GPT acts as an Application Developer Expert in python using streamlit. Build the Machine Learning Application with the Following workflow:
2. Begins by creating a button of Greetings lets take an Example we create a button of Greeting and when the user press the button then the baloons poopup and say hi, hello there ain some unique and attractive way also provide some brief discription of the Application within the button.
3. Ask the user if he wants to upload the Data or wants to use the Example Data
then create a button to ask the user to upload the Dataset if the user wants to uploads his own dataset create this button as a sidebar to upload the Data in the form of csv, xlsx ,tsv or any other file format.
4. if the user donot want to upload the Dataset then provide the default dataset Button or selectionbox as a sidebar that load the Dataset from seaborn libarary using sns.loaddataset()function.The Dataset should include titanic,tips and iris dataset.
5. print the basic information about the Dataset such as about the context and discription of dataset and also print the columns name , shape of the dataset , info and summary statistics and someother which is necessary in developing the Machine Learning App 
6. Ask from the user to select the columns as the Features and also teh columns as the target variables 
7. then if the user selects the continuous numeric columns then print that it is a   regression problem and if the user selects the categorical columns then print that it is a classification problem
8. Then preprocess the Data if the Missing values are present in the Data then fill these missing values with the iterative imputer if the missing values are in the percentage less than 50% and if the missing values percentage are greater tahn 80% or 90% then drop these columns 
9. Scale the columns which are not on the same scale i-e: if some values of the column are in the very low range and some are in very high range then scale them by using the standard scaler 
10. Encode the categorical columns in such a way that make the separate encoder for each column and encode them and then inverse transform each of them one by one at the last step of the application
11. Ask the user to select the train test split size via slider or user input function
12. Ask the user to select the model from the sidebar from the linearregression, support vector regressor, decision tree regressor and random forest regressor for the regression task if the user select the regression problem 
13. otherwise select the classification models such as support vector classifier, logistic classifier, decission tree classifier or random forest classifier for classification problem
14. Train the Model on the training Data and then evaluate on the test data
15. if the user select teh regression problem and selct the regression model then evaluate the result on the basis of regression metrics such as MSE,RMSE,MAE,r2_score and AUCROC curve and if the user selects the classification problem and also the classification model then print the result based on the classification evaluation metrics such as the Accuracy,precision , recall score, f1 score and also draw confusinmetrics based on it
16. print the evaluation metrics for each model of regression if the problem is of regression otherwise if the problem is of classification then print the classification evaluation mwtrics
17. Ask from the user to if he wants to predict the result then predict the results by providing the input data or as a slider or if he wants to predict the result of example data then provides him the orediction result of the example data after model training and then the fine tunning of the model nd choose the best model out of them
18. then ask the user if he wants to save and download the file then he can save the file in the form of pickle file


