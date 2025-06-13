# ###########################################################################
# # evaluation-script.R                                                     #
# #                                                                         #
# # Comparing Anomaly Detectors for Keystroke Biometrics                    #
# # Evaluation Proceedure                                                   #
# # R Script                                                                #
# #                                                                         #
# # by: Kevin Killourhy                                                     #
# # date: May 19, 2009                                                      #
# ###########################################################################

# library( MASS );
# library( ROCR );
# library( stats );


# # The euclideanTrain and euclideanScore functions comprise the
# # Euclidean anomaly detector.  During training, the detector takes a
# # set of password-timing vectors (encoded as rows in a matrix) and
# # calculates the mean vector.  This mean vector is returned as the
# # detection model.  During scoring, the detector takes the detection
# # model and a new set of password-timing vectors (also encoded as rows
# # in a matrix) and calculates the squared Euclidean distance between
# # the mean vector and each of the new password-timing vectors.  These
# # scores are returned in a vector whose length is equal to the number
# # of password-timing vectors in the scoring matrix.
# #euclideanScore <- function( detection.model, YScore ) {
# #  p <- length( detection.model ); #p <- length( detection.model$mean );
# #  n <- nrow( YScore );
# #  print( paste("Number of columns: ", p, "  Number of rows: ", n, sep = " ") );

# #  if( ncol(YScore) != p ) stop("Training/test feature length mismatch ");
  
# #  meanMatrix <- matrix( detection.model, byrow=TRUE, nrow=n, ncol=p );
  
# #  scores <- rowSums( ( YScore - detection.model )^2 );

# #  return( scores );
# #}

# # The mahalanobisTrain and mahalanobisScore functions comprise the
# # Mahalanobis anomaly detector.  During training, the detector takes a
# # set of password-timing vectors (encoded as rows in a matrix) and
# # calculates the mean vector and also the inverse of the covariance
# # matrix.  This vector and matrix are returned as the detection model.
# # During scoring, the detector takes the detection model and a new set
# # of password-timing vectors (also encoded as rows in a matrix) and
# # calculates the squared Mahalanobis distance between the mean vector
# # and each of the new password-timing vectors.  These scores are
# # returned in a vector whose length is equal to the number of
# # password-timing vectors in the scoring matrix.

# mahalanobisScore <- function( detection.model, YScore ) {
#   p <- length( detection.model$mean );
#   n <- nrow( YScore );

#   if (any(is.na(YScore))) stop("YScore содержит NA")
#   if (any(is.na(detection.model$mean))) stop("model$mean содержит NA")
#   if (any(is.na(detection.model$covInv))) stop("model$covInv содержит NA")

#   if( ncol(YScore) != p ) stop("Training/test feature length mismatch ");
  
#   scores <- mahalanobis( 
#     YScore,
#     detection.model$mean,
#     detection.model$covInv,
#     inverted=TRUE 
#   );
#   return( scores );
# }

# # Load in the training model
# detection.model.file <- '/Users/vadimnaumov/Desktop/learning/8_sem/NIR/keystroke_django_auth/test_model/R-scripts/dmod';
# if( ! file.exists(detection.model.file) ) {
#     stop( "Detection model file ", detection.model.file, " does not exist");
# }
# # detection.model <- unserialize( 
# #   charToRaw(readChar( 
# #     detection.model.file, 
# #     file.info(detection.model.file)$size ) 
# #   ) 
# # )

# detection.model <- readRDS(detection.model.file)

# # serialized_text <- readLines("/var/www/html/r/dmod")
# # detection.model <- eval(parse(text = serialized_text))

# # Load in "this" attempt's timing array
# current.attempt.file <- '~/Desktop/learning/8_sem/NIR/keystroke_django_auth/test_model/datasets/current_attempt.csv';
# if( ! file.exists(current.attempt.file) ) {
#     stop( "Current attempt data file ", current.attempt.file, " does not exist");
# }
# YScore <- read.csv( current.attempt.file,
#                     nrows=2,
#                     header=TRUE,
#                     stringsAsFactors=FALSE );


# # Drop the columns/rows from our data related to the Enter key's time up
# # (this is not recorded reliably by our Javascript)
# #length.with.Enters <- length( detection.model$mean );
# #length.new <- length.with.Enters - 2
# #detection.model$mean <- detection.model$mean[(1:length.new)]
# #YScore <- YScore[,(1:length.new)]
# #detection.model$covInv <- detection.model$covInv[(1:length.new),(1:length.new)]


# # Make the current attempt data a matrix
# YScore <- as.matrix( YScore )

# # Get the "score" (distance between the model and this attempt
# score <- mahalanobisScore( detection.model, YScore );

# # Scale the score based on the number of keys in the input
# # This is the per-key average deviation from the model (squared)
# deviation.avg <- score / length( detection.model );

# # An arbitrarily chosen maximum acceptable average deviation
# # Chosen because this is about double my worst score
# deviation.max <- 7000;

# prob.imposter <- deviation.avg / deviation.max
# if( prob.imposter > 1.0 ) {
#   prob.imposter <- 1.0
# }

# # Return to PHP
# write( "Probability you are an imposter:", "" );
# write( score, "" )

library(MASS)

# Загрузка обученной модели
detection.model.file <- "R-scripts/dmod"
if (!file.exists(detection.model.file)) {
    stop("Detection model file does not exist")
}
detection.model <- readRDS(detection.model.file)

# Загрузка всех попыток из общего CSV
YScore <- read.csv("datasets/current_attempt.csv", header=TRUE, stringsAsFactors=FALSE)
YScore <- as.matrix(YScore)

# Проверка размерности
if (ncol(YScore) != length(detection.model$mean)) {
    stop("Mismatch between model and test data dimensions")
}

# Расчёт отклонения
score <- mahalanobis(
  YScore,
  detection.model$mean,
  detection.model$covInv,
  inverted=TRUE
)

# # Return to PHP
# write( "Probability you are an imposter:", "" );
# write( score, "" )


# Среднее квадратичное отклонение на один элемент
deviation.avg <- score / length(detection.model)

# Максимальное допустимое значение
deviation.max <- 7000
prob.imposter <- deviation.avg / deviation.max
prob.imposter[prob.imposter > 1.0] <- 1.0

# Вывод строкой (по одному значению в строке)
for (p in score) {
  cat(p, "\n")
}