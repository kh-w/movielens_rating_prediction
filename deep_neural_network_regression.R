##########################################################################################
# Create edx set, validation set (final hold-out test set)
##########################################################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(keras)) install.packages("keras", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(lubridate)
library(stringr)
library(tidyr)
library(ggplot2)
library(gridExtra)
library(keras)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

########### Long runtime ###########
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
########### Long runtime ###########

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
gc()

memory.limit(size=15000)
options(digits = 5)

##########################################################################################

## feature 1 (movie popularity)

rating_by_popularity <- edx %>% 
  group_by(movieId) %>% 
  summarize(n_movie = n(), 
            moviemean = mean(rating))

edx <- edx %>% 
  left_join(rating_by_popularity %>% dplyr::select(-moviemean), by = "movieId")
validation <- validation %>% 
  left_join(rating_by_popularity %>% dplyr::select(-moviemean), by = "movieId")

##########################################################################################

## feature 2 (frequent rater)

rating_by_freqrater <- edx %>% 
  group_by(userId) %>% 
  summarize(n_user = n(), 
            usermean = mean(rating))

edx <- edx %>% 
  left_join(rating_by_freqrater %>% dplyr::select(-usermean), by = "userId")
validation <- validation %>% 
  left_join(rating_by_freqrater %>% dplyr::select(-usermean), by = "userId")

##########################################################################################

## feature 3 and 4 (movie mean and std error)

movie_mean <- edx %>% 
  group_by(movieId) %>% 
  summarize(movieMean = mean(rating), 
            movieMeanStdError = sd(rating)/sqrt(n()))

edx <- edx %>% 
  left_join(movie_mean, by = "movieId") %>% 
  mutate(movieMeanStdError = replace_na(movieMeanStdError, 5))
validation <- validation %>% 
  left_join(movie_mean, by = "movieId") %>% 
  mutate(movieMeanStdError = replace_na(movieMeanStdError, 5))

##########################################################################################

## feature 5 and 6 (user mean and std error)

user_mean <- edx %>% 
  group_by(userId) %>% 
  summarize(userMean = mean(rating), 
            userMeanStdError = sd(rating)/sqrt(n()))

edx <- edx %>% 
  left_join(user_mean, by = "userId") %>% 
  mutate(userMeanStdError = replace_na(userMeanStdError, 5))
validation <- validation %>% 
  left_join(user_mean, by = "userId") %>% 
  mutate(userMeanStdError = replace_na(userMeanStdError, 5))

##########################################################################################

## feature 7 to 12 (movie year, rate year, rate month, rate day, rate weekday, rate hour)

edx <- edx %>% mutate(movieYear = as.numeric(str_replace(str_replace(str_extract(title, "\\([0-9]{4}\\)"),"\\(",""),"\\)","")), 
                      rateYear = year(as_datetime(timestamp)),
                      rateMonth = month(as_datetime(timestamp)),
                      rateDay = day(as_datetime(timestamp)),
                      rateWday = wday(as_datetime(timestamp)),
                      rateHour = hour(as_datetime(timestamp)))

validation <- validation %>% mutate(movieYear = as.numeric(str_replace(str_replace(str_extract(title, "\\([0-9]{4}\\)"),"\\(",""),"\\)","")), 
                                    rateYear = year(as_datetime(timestamp)),
                                    rateMonth = month(as_datetime(timestamp)),
                                    rateDay = day(as_datetime(timestamp)),
                                    rateWday = wday(as_datetime(timestamp)),
                                    rateHour = hour(as_datetime(timestamp)))

##########################################################################################

## feature 13 and 14 (genre mean and std error)

genre_rating_by_year <- edx %>%
  group_by(genres) %>%
  summarize(n = n(), 
            genreMean = mean(rating), 
            genreMeanStdError = sd(rating)/sqrt(n)) %>%
  filter(n > 0.0025*9000055) %>%
  dplyr::select(-n)

edx <- edx %>% 
  left_join(genre_rating_by_year, by = "genres") %>% 
  mutate(genreMean = replace_na(genreMean, 5), 
         genreMeanStdError = replace_na(genreMeanStdError, 5))
validation <- validation %>% 
  left_join(genre_rating_by_year, by = "genres") %>% 
  mutate(genreMean = replace_na(genreMean, 5), 
         genreMeanStdError = replace_na(genreMeanStdError, 5))

##########################################################################################

## feature 15 (user sd)

user_sd <- edx %>% 
  group_by(userId) %>% 
  summarize(userSD = sd(rating))

edx <- edx %>% 
  left_join(user_sd, by = "userId") %>% 
  mutate(userSD = replace_na(userSD, 5))
validation <- validation %>% 
  left_join(user_sd, by = "userId") %>% 
  mutate(userSD = replace_na(userSD, 5))

##########################################################################################

## feature 16 (movie sd)

movie_sd <- edx %>% 
  group_by(movieId) %>% 
  summarize(movieSD = sd(rating))

edx <- edx %>% 
  left_join(movie_sd, by = "movieId") %>% 
  mutate(movieSD = replace_na(movieSD, 5))
validation <- validation %>% 
  left_join(movie_sd, by = "movieId") %>% 
  mutate(movieSD = replace_na(movieSD, 5))

##########################################################################################

## feature 17 (title length)

shorttitle <- function(string){
  str_trunc(string, str_locate(string,"\\(")[1]+2, "right")
}

titles <- data.frame(title=unique(edx[,"title"])) %>% mutate(titleshort = unlist(lapply(title, shorttitle)), titlewords = str_count(titleshort,"( )"))

titlelen_mean <- edx[,1:6] %>% 
  left_join(titles[,c("title","titlewords")], by = "title") %>%
  group_by(titlewords) %>% 
  summarize(n = n(), mean = mean(rating), stderr = sd(rating)/sqrt(n)) 

edx <- edx %>% 
  left_join(titles[,c("title","titlewords")], by = "title")
validation <- validation %>% 
  left_join(titles[,c("title","titlewords")], by = "title")

##########################################################################################

summary(edx)

##########################################################################################

movielens <- c()
movielens$train$x <- edx[,c("n_movie","n_user","movieMean","movieMeanStdError","userMean","userMeanStdError",
                            "movieYear","rateYear","rateMonth","rateDay","rateWday","rateHour","genreMean",
                            "genreMeanStdError","userSD","movieSD","titlewords")]
movielens$train$x <- array(unlist(movielens$train$x), dim=dim(movielens$train$x))
dim(movielens$train$x)
movielens$test$x <- validation[,c("n_movie","n_user","movieMean","movieMeanStdError","userMean","userMeanStdError",
                                  "movieYear","rateYear","rateMonth","rateDay","rateWday","rateHour","genreMean",
                                  "genreMeanStdError","userSD","movieSD","titlewords")]
movielens$test$x <- array(unlist(movielens$test$x), dim=dim(movielens$test$x))
dim(movielens$test$x)
movielens$train$y <- edx[,"rating"]
movielens$train$y <- unlist(movielens$train$y)
dim(movielens$train$y)
movielens$test$y <- validation[,"rating"]
movielens$test$y <- unlist(movielens$test$y)
##########################################################################################

# normalize the data

mean <- apply(movielens$train$x, 2, mean)
sd <- apply(movielens$train$x, 2, sd)
movielens$train$x <- scale(movielens$train$x, center=mean, scale=sd)
movielens$test$x <- scale(movielens$test$x, center=mean, scale=sd)

input_dim <- dim(movielens$train$x)[2]

##########################################################################################

# ANN

callmodel <- function(){
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = input_dim) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)
  model %>% compile(optimizer = "rmsprop",
                    loss = "mse",
                    metrics = "mae")
}

##########################################################################################

model <- callmodel()

model %>% fit(movielens$train$x, 
              movielens$train$y, 
              validation_data = list(movielens$test$x, 
                                     movielens$test$y),
              epochs = 20, batch_size = 10000)

results <- model %>% evaluate(movielens$test$x, movielens$test$y)
