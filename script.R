######################################################################
##################### MOVIE LENS PROJECT #############################
######################################################################



###################################
#### LOADING THE REQUIRED PACKAGES
###################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(ggthemes)
library(lubridate)
options(digits = 4)
options(scipen = 999)



###################
#### LOADING DATA
###################

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

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



#######################
#### DATA EXPLORATION
#######################

str(edx)            #structure of edx dataset
str(validation)     #structure of validation dataset

summary(edx)        #summary of edx dataset

length(unique(edx$movieId))    #number of unique movies in edx dataset
length(unique(edx$userId))     #number of unique users in edx dataset


## Most popular movies in edx dataset:
edx %>% group_by(title) %>%
  summarise(ratings=n()) %>%
  arrange(desc(ratings)) %>%
  top_n(10, ratings) %>%
  ggplot(aes(ratings, reorder(title,ratings))) + 
  geom_col() +
  xlab("Count") + ylab("Title") +
  theme_calc()


## Most popular genres in edx dataset:
edx %>% group_by(genres) %>% 
  summarize(n=n()) %>% 
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>% 
  summarize(count=sum(n)) %>% 
  ggplot(aes(count, reorder(genres,count))) + 
  geom_col() +
  xlab("Count") + ylab("Genre") +
  theme_calc()


## Distribution of ratings in edx dataset:
min(edx$rating) #minimum rating in edx dataset
max(edx$rating) #maximum rating in edx dataset

ggplot(edx, aes(rating)) + 
  geom_histogram(binwidth = 0.25) +
  xlab("Rating") + ylab("Count") +
  theme_calc()



##############
#### METHODS
#############

## Generating training set and test set

set.seed(1)

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, 
                                  list = FALSE)
train <- edx[-test_index,]
test <- edx[test_index,]

rm(test_index)


## Making sure there are no movies nor users in the test set
## that don't appear on the training set

test <- test %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")  


## Loss function

RMSE <- function(true, prediction){
  sqrt(mean((true - prediction)^2))
}


## First method: Average rating
mu <- mean(train$rating) # expected value

rmse_avg <- RMSE(mu, test$rating)

rmse_results <- tibble(method = "Average rating",
                       RMSE = rmse_avg)

rmse_results


## Second method: Incorporating MOVIE EFFECTS
avgs_movie <- train %>% group_by(movieId) %>% 
  summarize(bi = mean(rating - mu)) # movie effects

pred_movies <- test %>%
  left_join(avgs_movie, by = "movieId") %>%
  mutate(pred_movies = mu+bi) %>% 
  pull(pred_movies)

rmse_movies <- RMSE(pred_movies, test$rating)

rmse_results <- rmse_results %>% rbind(c("Movie effects",
                                         rmse_movies))
rmse_results


## Third method: Incorporating MOVIE + USER EFFECTS
avgs_user <- train %>%
  left_join(avgs_movie, by='movieId') %>%
  group_by(userId) %>%
  summarize(bu = mean(rating - mu - bi)) # user effects

pred_users <- test %>% 
  left_join(avgs_movie, by='movieId') %>%
  left_join(avgs_user, by='userId') %>%
  mutate(pred_users = mu + bi + bu) %>%
  pull(pred_users)

rmse_users <- RMSE(pred_users, test$rating)

rmse_results <- rmse_results %>% rbind(c("Movie + User effects",
                                         rmse_users))

rmse_results


## Fourth method: Incorporating MOVIE + USER + GENRE EFFECTS

avgs_genre <- train %>%
  left_join(avgs_movie, by='movieId') %>%
  left_join(avgs_user, by='userId') %>%
  group_by(genres) %>%
  summarize(bg = mean(rating - mu - bi - bu)) # genre effects

pred_genres <- test %>% 
  left_join(avgs_movie, by='movieId') %>%
  left_join(avgs_user, by='userId') %>%
  left_join(avgs_genre, by='genres') %>%
  mutate(pred_genres = mu + bi + bu + bg) %>%
  pull(pred_genres)

rmse_genres <- RMSE(pred_genres, test$rating)

rmse_results <- rmse_results %>% rbind(c("Movie + User + Genre effects",
                                         rmse_genres))
rmse_results




## Fifth Method: Incorporating MOVIE + USER + GENRE + YEAR EFFECTS

train <- train %>% mutate(y = as.numeric(str_sub(title,-5,-2)))

test <- test %>% mutate(y = as.numeric(str_sub(title,-5,-2)))

avgs_year <- train %>%
  left_join(avgs_movie, by='movieId') %>%
  left_join(avgs_user, by='userId') %>%
  left_join(avgs_genre, by='genres') %>%
  group_by(y) %>%
  summarize(by = mean(rating - mu - bi - bu - bg)) # year effects

pred_year <- test %>% 
  left_join(avgs_movie, by='movieId') %>%
  left_join(avgs_user, by='userId') %>%
  left_join(avgs_genre, by='genres') %>%
  left_join(avgs_year, by='y') %>%
  mutate(pred_year = mu + bi + bu + bg + by) %>%
  pull(pred_year)

rmse_year <- RMSE(pred_year, test$rating)

rmse_results <- rmse_results %>% rbind(c("Movie + User + Genre + Year effects",
                                         rmse_year))
rmse_results




## Sixth Method: Incorporating MOVIE + USER + GENRE + YEAR + MONTH EFFECTS

train <- train %>% mutate(m = as.numeric(month(as_datetime(timestamp))))

test <- test %>% mutate(m = as.numeric(month(as_datetime(timestamp))))

avgs_month <- train %>%
  left_join(avgs_movie, by='movieId') %>%
  left_join(avgs_user, by='userId') %>%
  left_join(avgs_genre, by='genres') %>%
  left_join(avgs_year, by = 'y') %>%
  group_by(m) %>%
  summarize(bm = mean(rating - mu - bi - bu - bg - by)) # month effects

pred_month <- test %>% 
  left_join(avgs_movie, by='movieId') %>%
  left_join(avgs_user, by='userId') %>%
  left_join(avgs_genre, by='genres') %>%
  left_join(avgs_year, by='y') %>%
  left_join(avgs_month, by='m') %>%
  mutate(pred_month = mu + bi + bu + bg + by + bm) %>%
  pull(pred_month)

rmse_month <- RMSE(pred_month, test$rating)

rmse_results <- rmse_results %>% rbind(c("Movie + User + Genre + Year + Month effects",
                                         rmse_month))
rmse_results




## Seventh Method: REGULARIZATION

## Choosing lambda by Cross Validation

lambdas <- seq(0, 10, 0.25)

rmses_reg <- sapply(lambdas, function(lambda){
  bi <- train %>%
    group_by(movieId) %>%
    summarize(bi = sum(rating - mu)/(n()+lambda))
  bu <- train %>%
    left_join(bi, by="movieId") %>%
    group_by(userId) %>%
    summarize(bu = sum(rating - mu - bi)/(n()+lambda))
  bg <- train %>%
    left_join(bi, by="movieId") %>%
    left_join(bu, by="userId") %>%
    group_by(genres) %>%
    summarize(bg = sum(rating - mu - bi - bu)/(n()+lambda))
  by <- train %>%
    left_join(bi, by="movieId") %>%
    left_join(bu, by="userId") %>%
    left_join(bg, by="genres") %>%
    group_by(y) %>%
    summarize(by = sum(rating - mu - bi - bu - bg)/(n()+lambda))
  bm <- train %>%
    left_join(bi, by="movieId") %>%
    left_join(bu, by="userId") %>%
    left_join(bg, by="genres") %>%
    left_join(by, by="y") %>%
    group_by(m) %>%
    summarize(bm = sum(rating - mu - bi - bu - bg - by)/(n()+lambda))
  
  predicted_ratings <-
    test %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    left_join(bg, by = "genres") %>%
    left_join(by, by = "y") %>%
    left_join(bm, by = "m") %>%
    mutate(pred = mu + bi + bu + bg + by + bm) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test$rating))
})

qplot(lambdas, rmses_reg)

lambda <- lambdas[which.min(rmses_reg)]
lambda


## Predictions using regularization

avgs_moviereg <- train %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(n()+lambda))

avgs_userreg <- train %>%
  left_join(avgs_moviereg, by="movieId") %>%
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(n()+lambda))

avgs_genrereg <- train %>%
  left_join(avgs_moviereg, by="movieId") %>%
  left_join(avgs_userreg, by="userId") %>%
  group_by(genres) %>%
  summarize(bg = sum(rating - mu - bi - bu)/(n()+lambda))

avgs_yearreg <- train %>%
  left_join(avgs_moviereg, by='movieId') %>%
  left_join(avgs_userreg, by='userId') %>%
  left_join(avgs_genrereg, by='genres') %>%
  group_by(y) %>%
  summarize(by = sum(rating - mu - bi - bu - bg)/(n()+lambda))

avgs_monthreg <- train %>%
  left_join(avgs_moviereg, by='movieId') %>%
  left_join(avgs_userreg, by='userId') %>%
  left_join(avgs_genrereg, by='genres') %>%
  left_join(avgs_yearreg, by = 'y') %>%
  group_by(m) %>%
  summarize(bm = sum(rating - mu - bi - bu - bg - by)/(n()+lambda)) # month effects

pred_reg <- test %>%
  left_join(avgs_moviereg, by = "movieId") %>%
  left_join(avgs_userreg, by = "userId") %>%
  left_join(avgs_genrereg, by = "genres") %>%
  left_join(avgs_yearreg, by = "y") %>%
  left_join(avgs_monthreg, by = "m") %>%
  mutate(pred = mu + bi + bu + bg + by + bm) %>%
  pull(pred)

rmse_reg <- RMSE(pred_reg, test$rating)

rmse_results <- rmse_results %>% rbind(c("Regularization",
                                         rmse_reg))
rmse_results

## MAXMIN

min(pred_reg) ## -0.5117
max(pred_reg) ## 6.02
max(train$rating) ## 5
min(train$rating) ## 0.5

pred_maxmin <- test %>%
  left_join(avgs_moviereg, by = "movieId") %>%
  left_join(avgs_userreg, by = "userId") %>%
  left_join(avgs_genrereg, by = "genres") %>%
  left_join(avgs_yearreg, by = "y") %>%
  left_join(avgs_monthreg, by = "m") %>%
  mutate(pred = mu + bi + bu + bg + by + bm) %>%
  mutate(pred_maxmin = ifelse(pred>5,5,ifelse(pred<0.5,0.5,pred))) %>%
  pull(pred_maxmin)

rmse_maxmin <- RMSE(pred_maxmin, test$rating)

rmse_results <- rmse_results %>% rbind(c("Adjusting max and min values",
                                         rmse_maxmin))
rmse_results

## FUNCTION TO CHECK SCORE
score <- function(x){
  ifelse(x>=0.9, "5 points",
         ifelse(x>=0.86550, "10 points",
                ifelse(x>=0.86500, "15 points",
                       ifelse(x>=0.86490, "20 points",
                              "25 points"))))
}

rmse_results <- rmse_results %>% mutate(score=score(RMSE))
rmse_results


## EVALUATION USING VALIDATION SET

edx <- edx %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
validation <- validation %>% mutate(year = as.numeric(str_sub(title,-5,-2)))

edx <- edx %>% mutate(month = as.numeric(month(as_datetime(timestamp))))
validation <- validation %>% mutate(month = as.numeric(month(as_datetime(timestamp))))

edx_avgs_moviereg <- edx %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(n()+lambda))

edx_avgs_userreg <- edx %>%
  left_join(edx_avgs_moviereg, by="movieId") %>%
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(n()+lambda))

edx_avgs_genrereg <- edx %>%
  left_join(edx_avgs_moviereg, by="movieId") %>%
  left_join(edx_avgs_userreg, by="userId") %>%
  group_by(genres) %>%
  summarize(bg = sum(rating - mu - bi - bu)/(n()+lambda))

edx_avgs_yearreg <- edx %>%
  left_join(edx_avgs_moviereg, by="movieId") %>%
  left_join(edx_avgs_userreg, by="userId") %>%
  left_join(edx_avgs_genrereg, by="genres") %>%
  group_by(year) %>%
  summarize(by = sum(rating - mu - bi - bu - bg)/(n()+lambda))

edx_avgs_monthreg <- edx %>%
  left_join(edx_avgs_moviereg, by="movieId") %>%
  left_join(edx_avgs_userreg, by="userId") %>%
  left_join(edx_avgs_genrereg, by="genres") %>%
  left_join(edx_avgs_yearreg, by="year") %>%
  group_by(month) %>%
  summarize(bm = sum(rating - mu - bi - bu - by)/(n()+lambda))

edx_pred <- validation %>%
  left_join(edx_avgs_moviereg, by = "movieId") %>%
  left_join(edx_avgs_userreg, by = "userId") %>%
  left_join(edx_avgs_genrereg, by = "genres") %>%
  left_join(edx_avgs_yearreg, by = "year") %>%
  left_join(edx_avgs_monthreg, by = "month") %>%
  mutate(pred = mu + bi + bu + bg + by + bm) %>%
  mutate(pred_maxmin = ifelse(pred>5,5,ifelse(pred<0.5,0.5,pred))) %>%
  pull(pred_maxmin)

RMSE(edx_pred, validation$rating)

score(RMSE(edx_pred, validation$rating))






























## EVALUATION USING VALIDATION SET

lambda <- 4.75
mu <- mean(edx$rating)

edx_avgs_moviereg <- edx %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(n()+lambda))

edx_avgs_userreg <- train %>%
  left_join(edx_avgs_moviereg, by="movieId") %>%
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(n()+lambda))

edx_avgs_genrereg <- train %>%
  left_join(edx_avgs_moviereg, by="movieId") %>%
  left_join(edx_avgs_userreg, by="userId") %>%
  group_by(genres) %>%
  summarize(bg = sum(rating - mu - bi - bu)/(n()+lambda))

edx_avgs_yearreg <- train %>%
  left_join(edx_avgs_moviereg, by="movieId") %>%
  left_join(edx_avgs_userreg, by="userId") %>%
  left_join(edx_avgs_genrereg, by="genres") %>%
  group_by(year) %>%
  summarize(by = sum(rating - mu - bi - bu - bg)/(n()+lambda))

edx_pred <- validation %>%
  left_join(edx_avgs_moviereg, by = "movieId") %>%
  left_join(edx_avgs_userreg, by = "userId") %>%
  left_join(edx_avgs_genrereg, by = "genres") %>%
  left_join(edx_avgs_yearreg, by = "year") %>%
  mutate(pred = mu + bi + bu + bg + by) %>%
  mutate(pred_maxmin = ifelse(pred>5,5,ifelse(pred<0.5,0.5,pred))) %>%
  pull(pred_maxmin)

RMSE(edx_pred, validation$rating)











library(lubridate)
ymd_hms(edx$timestamp)

edx <- edx %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
validation <- validation %>% mutate(year = as.numeric(str_sub(title,-5,-2)))

edx <- edx %>% mutate(month = as.numeric(month(as_datetime(timestamp))))
validation <- validation %>% mutate(month = as.numeric(month(as_datetime(timestamp))))





lambda <- 4.75
mu <- mean(edx$rating)

edx_avgs_moviereg <- edx %>%
  group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(n()+lambda))

edx_avgs_userreg <- edx %>%
  left_join(edx_avgs_moviereg, by="movieId") %>%
  group_by(userId) %>%
  summarize(bu = sum(rating - mu - bi)/(n()+lambda))

edx_avgs_genrereg <- edx %>%
  left_join(edx_avgs_moviereg, by="movieId") %>%
  left_join(edx_avgs_userreg, by="userId") %>%
  group_by(genres) %>%
  summarize(bg = sum(rating - mu - bi - bu)/(n()+lambda))

edx_avgs_yearreg <- edx %>%
  left_join(edx_avgs_moviereg, by="movieId") %>%
  left_join(edx_avgs_userreg, by="userId") %>%
  left_join(edx_avgs_genrereg, by="genres") %>%
  group_by(year) %>%
  summarize(by = sum(rating - mu - bi - bu - bg)/(n()+lambda))

edx_avgs_monthreg <- edx %>%
  left_join(edx_avgs_moviereg, by="movieId") %>%
  left_join(edx_avgs_userreg, by="userId") %>%
  left_join(edx_avgs_genrereg, by="genres") %>%
  left_join(edx_avgs_yearreg, by="year") %>%
  group_by(month) %>%
  summarize(bm = sum(rating - mu - bi - bu - bg - by)/(n()+lambda))

edx_pred <- validation %>%
  left_join(edx_avgs_moviereg, by = "movieId") %>%
  left_join(edx_avgs_userreg, by = "userId") %>%
  left_join(edx_avgs_genrereg, by = "genres") %>%
  left_join(edx_avgs_yearreg, by = "year") %>%
  left_join(edx_avgs_monthreg, by = "month") %>%
  mutate(pred = mu + bi + bu + bg + by + bm) %>%
  mutate(pred_maxmin = ifelse(pred>5,5,ifelse(pred<0.5,0.5,pred))) %>%
  pull(pred_maxmin)

RMSE(edx_pred, validation$rating)
