##########################################################################################
# Create edx set, validation set (final hold-out test set)
##########################################################################################
tic <- Sys.time()
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

library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(lubridate)
library(stringr)
library(tidyr)
library(ggplot2)
library(gridExtra)

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

## plot the average rating versus number of ratings
plot_moviepop <- rating_by_popularity %>% 
  ggplot(aes(n_movie, moviemean)) + 
  geom_smooth() +
  ggtitle("Average rating vs number of ratings") +
  labs(x = "Number of ratings", y = "Average rating")
##

edx <- edx %>% 
  left_join(rating_by_popularity %>% dplyr::select(-moviemean), by = "movieId")
validation <- validation %>% 
  left_join(rating_by_popularity %>% dplyr::select(-moviemean), by = "movieId")

gc()

##########################################################################################

## feature 2 (frequent rater)

rating_by_freqrater <- edx %>% 
  group_by(userId) %>% 
  summarize(n_user = n(), 
            usermean = mean(rating))

## plot the average rating versus user's number of rating
plot_rating_hist <- edx %>% 
  group_by(rating) %>% 
  summarize(n = n()) %>% 
  ggplot(aes(rating, n)) + 
  geom_bar(stat="identity") +
  ggtitle("Frequency of each rating")
plot_freqrater <- rating_by_freqrater %>% 
  ggplot(aes(n_user, usermean)) + 
  geom_smooth() +
  ggtitle("Average rating vs user's number of rating") +
  labs(x = "User's number of ratings", y = "Average rating")
##

edx <- edx %>% 
  left_join(rating_by_freqrater %>% dplyr::select(-usermean), by = "userId")
validation <- validation %>% 
  left_join(rating_by_freqrater %>% dplyr::select(-usermean), by = "userId")

gc()

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

gc()

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

gc()

##########################################################################################

## feature 7 to 12 (movie year, rate year, rate month, rate day, rate weekday, rate hour)

edx <- edx %>% mutate(movieYear = as.numeric(str_replace(str_replace(str_extract(title, "\\([0-9]{4}\\)"),"\\(",""),"\\)","")), 
                      rateYear = year(as_datetime(timestamp)),
                      rateMonth = month(as_datetime(timestamp)),
                      rateDay = day(as_datetime(timestamp)),
                      rateWday = wday(as_datetime(timestamp)),
                      rateHour = hour(as_datetime(timestamp)))

## plot the ratings against the date fields
plot_movieyear <- edx %>% 
  group_by(movieYear) %>% 
  summarize(movieYearMean = mean(rating), 
            movieYearMeanStdError = sd(rating)/sqrt(n())) %>%
  mutate(movieYear = reorder(movieYear, movieYearMean)) %>%
  ggplot(aes(x = movieYear, y = movieYearMean, ymin = movieYearMean - 2*movieYearMeanStdError, ymax = movieYearMean + 2*movieYearMeanStdError)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Average rating vs movie release year") +
  labs(x = "Movie release year", y = "Average rating")
plot_rateyear <- edx %>% 
  group_by(rateYear) %>% 
  summarize(rateYearMean = mean(rating), 
            rateYearMeanStdError = sd(rating)/sqrt(n())) %>%
  filter(rateYear != 1995) %>%
  ggplot(aes(x = rateYear, y = rateYearMean, ymin = rateYearMean - 2*rateYearMeanStdError, ymax = rateYearMean + 2*rateYearMeanStdError)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Average rating vs year of the rating") +
  labs(x = "Year of the rating", y = "Average rating")
plot_ratemonth <- edx %>% 
  group_by(rateMonth) %>% 
  summarize(rateMonthMean = mean(rating), 
            rateMonthMeanStdError = sd(rating)/sqrt(n())) %>%
  ggplot(aes(x = rateMonth, y = rateMonthMean, ymin = rateMonthMean - 2*rateMonthMeanStdError, ymax = rateMonthMean + 2*rateMonthMeanStdError)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Average rating vs month of the rating") +
  labs(x = "Month of the rating", y = "Average rating")
plot_rateday <- edx %>% 
  group_by(rateDay) %>% 
  summarize(rateDayMean = mean(rating), 
            rateDayMeanStdError = sd(rating)/sqrt(n())) %>%
  ggplot(aes(x = rateDay, y = rateDayMean, ymin = rateDayMean - 2*rateDayMeanStdError, ymax = rateDayMean + 2*rateDayMeanStdError)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Average rating vs day of the rating") +
  labs(x = "Day of the rating", y = "Average rating")
plot_rateWday <- edx %>% 
  group_by(rateWday) %>% 
  summarize(rateWdayMean = mean(rating), 
            rateWdayMeanStdError = sd(rating)/sqrt(n())) %>%
  ggplot(aes(x = rateWday, y = rateWdayMean, ymin = rateWdayMean - 2*rateWdayMeanStdError, ymax = rateWdayMean + 2*rateWdayMeanStdError)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Average rating vs weekday of the rating") +
  labs(x = "Weekday of the rating", y = "Average rating")
plot_ratehour <- edx %>% 
  group_by(rateHour) %>% 
  summarize(rateHourMean = mean(rating), 
            rateHourMeanStdError = sd(rating)/sqrt(n())) %>%
  ggplot(aes(x = rateHour, y = rateHourMean, ymin = rateHourMean - 2*rateHourMeanStdError, ymax = rateHourMean + 2*rateHourMeanStdError)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Average rating vs hour of the rating") +
  labs(x = "Hour of the rating", y = "Average rating")

plot_movieyear
grid.arrange(grobs = list(plot_rateyear, plot_ratemonth, plot_rateday, plot_rateWday, plot_ratehour), ncol = 3)

##

validation <- validation %>% mutate(movieYear = as.numeric(str_replace(str_replace(str_extract(title, "\\([0-9]{4}\\)"),"\\(",""),"\\)","")), 
                                    rateYear = year(as_datetime(timestamp)),
                                    rateMonth = month(as_datetime(timestamp)),
                                    rateDay = day(as_datetime(timestamp)),
                                    rateWday = wday(as_datetime(timestamp)),
                                    rateHour = hour(as_datetime(timestamp)))

gc()

##########################################################################################

## feature 13 and 14 (genre mean and std error)

## plot the average rating versus genre 
plot_genre <- edx %>% group_by(genres) %>%
  summarize(n = n(), 
            genreMean = mean(rating), 
            genreMeanStdError = sd(rating)/sqrt(n())) %>%
  filter(n >= 0.0025*9000055) %>% 
  mutate(genres = reorder(genres, genreMean)) %>%
  ggplot(aes(x = genres, 
             y = genreMean, 
             ymin = genreMean - 2*genreMeanStdError, 
             ymax = genreMean + 2*genreMeanStdError)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Average rating vs genres") +
  labs(x = "Genre(s)", y = "Average rating")
##

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

gc()

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

gc()

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

gc()

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

## plot the average rating versus title length
plot_titlelen <- titlelen_mean %>%
  mutate(titlewords = reorder(titlewords, mean)) %>%
  ggplot(aes(x = titlewords, y = mean, ymin = mean - 2*stderr, ymax = mean + 2*stderr)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Average rating vs title length") +
  labs(x = "Title length", y = "Average rating")
##

edx <- edx %>% 
  left_join(titles[,c("title","titlewords")], by = "title")
validation <- validation %>% 
  left_join(titles[,c("title","titlewords")], by = "title")

gc()

##########################################################################################

## model fitting

fit.low <- lm(rating ~
                movieMean+
                userMean+
                n_movie,
               data = edx[,-c("userId","movieId","timestamp","title","genres")])
lmfitbest <- step(fit.low, scope = list(lower = rating ~
                                           movieMean+
                                           userMean+
                                           n_movie,
                                         upper = rating ~
                                                 n_movie+
                                                 n_user+
                                                 movieMean+
                                                 movieMeanStdError+
                                                 userMean+
                                                 userMeanStdError+
                                                 movieYear+
                                                 rateYear+
                                                 rateMonth+
                                                 rateDay+
                                                 rateWday+
                                                 rateHour+
                                                 genreMean+
                                                 genreMeanStdError+
                                                 userSD+
                                                 movieSD+
                                                 titlewords),
                  direction = "forward")
summary(lmfitbest)

##########################################################################################

bound <- function(x,a,b){max(min(x,b),a)}

##########################################################################################

## prediction on edx

edx$y_hat <- sapply(predict.lm(lmfitbest, newdata = edx),bound,a=0.5,b=5)

RMSE(edx$y_hat,edx$rating)

gc()

##########################################################################################

## prediction on validation 

validation$y_hat <- sapply(predict.lm(lmfitbest, newdata = validation),bound,a=0.5,b=5)

RMSE(validation$y_hat,validation$rating)

gc()

##########################################################################################

toc <- Sys.time()

toc - tic

##########################################################################################

save.image("movielens_ws.RData")
