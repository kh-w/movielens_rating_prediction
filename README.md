# Overview
In this project, a system is developed to predict users' rating on movies they have not watched before. The system is being trained by a dataset and it is obtained at <http://files.grouplens.org/datasets/movielens/ml-10m.zip>. The dataset consists of 6 columns: User ID, Movie ID, Movie Title, Genre, Rating, and Timestamp of the rating. 90% of this dataset (9,000,055 rows) is defined as the training dataset and is used for developing the system. The remaining 10% (999,999 rows) is defined as test dataset which is treated as new data for the final model. The prediction will be compared to the actual ratings in the test dataset and the root mean squared error (RMSE) will be reported. 

The methodology is to mine appropriate and useful data structures from the training dataset, transform these structures into predictors of the model. The model is a linear regression with the best model selected by forward selection.

The goal of this project is to recommend movies to the users, therefore the ultimate product is a list of recommendation with the most recommended movie being listed on the top of the list and followed by second recommended, and so on. 
