# ---- setup-functions ----

library(geosphere)

# Takes a comparison and a sigma value
# Returns the corresponding Gaussian kernel
gaussian_kernel <- function(comp, sigma) {
  numerator <- -as.numeric(comp)^2
  return (exp(numerator / (2 * sigma^2)))
}

# Calculates the difference in days between two days taking year wrap-around
# into account
calculate_day_diff <- function(a, b) {
  diff <- abs(a - b)
  return(pmin(diff, 365 - diff))
}

# Calculates the difference in hours between two timestamps in 24-hour format
calculate_hour_diff <- function(a, b) {
  diff <- abs(outer(a, b, "-"))
  return(pmin(diff, 24 - diff))
}

set.seed(1234567890)

stations <- read.csv("stations.csv", fileEncoding = "latin1")
temps <- read.csv("temps50k.csv")
st <- merge(stations,temps,by="station_number")

h_distance <- 0 # These three values are up to the students
h_date <- 0
h_time <- 0

times <- c("04:00:00", "06:00:00", "08:00:00", "10:00:00", "12:00:00",
           "14:00:00", "16:00:00", "18:00:00", "20:00:00", "22:00:00",
           "24:00:00")

temp <- vector(length=length(times))

# ---- setup-predictions ----

a <- 16.346865493338598 # The point to predict (up to the students)
b <- 56.66857193484579
pred_date <- "2000-10-04" # The date to predict (up to the students)

# ---- filter-data ----

# Filter out dates after the date to predict
st <- st[as.Date(st$date) < as.Date(pred_date), ]

# ---- calculate-physical-distance ----
dh <- distHaversine(st[, c("longitude", "latitude")], c(a, b))

# Uses the first quartile of the result of the Haversine function as sigma
# (approx. 233km)
h_distance <- gaussian_kernel(dh, quantile(dh, 0.25))

# ---- calculate-day-distance ----

# Calculate the difference in days
day_diff <- calculate_day_diff(
  as.numeric(format(as.Date(st$date), "%j")),
  as.numeric(format(as.Date(pred_date), "%j"))
)

# Uses the first quartile of the difference in days as sigma
# (45 days)
h_date <- gaussian_kernel(day_diff, quantile(day_diff, 0.25))

# ---- calculate-hour-distance ----

# 50 000 x 11 matrix with the time differences ranging from 0 - 12.
# each row represents an entry in st and each column represents a 2 hour
# interval in the span 04:00 - 24:00
hour_diff <- calculate_hour_diff(
  as.numeric(substr(st$time, 1, 2)), as.numeric(substr(times, 1, 2))
)

# 50 000 x 11 matrix with the values from the Gaussian kernel.
# each row represents an entry in st and each column represents a 2 hour
# interval in the span 04:00 - 24:00
h_time <- matrix(nrow = nrow(st), ncol = length(times))
for (i in 1:ncol(hour_diff)) {
  # Uses the first quartile of the difference in time for every 2nd hour between
  # 04:00 - 24:00 (values in range 2:4)
  h_time[, i] <- gaussian_kernel(hour_diff[, i], quantile(hour_diff[, i], 0.25))
}

# ---- . ----

# Convert distance and day vectors to matrices to allow them to be added with
# the time matrix
h_distance_m <- matrix(
  h_distance, nrow = length(h_distance), ncol = length(times)
)

h_date_m <- matrix(
  h_date, nrow = length(h_date), ncol = length(times)
)

# Sum of kernels
total_additive <- h_distance_m + h_date_m + h_time

# Make predictions using the given formula
predictions_additive <- apply(total_additive, 2,
                     function (x) sum(x * st$air_temperature) / sum(x))

print(predictions_additive)

# Multiplication of kernels
total_multiplicative <- h_distance_m * h_date_m * h_time

# Make predictions using the given formula
predictions_multiplicative <- apply(
  total_multiplicative, 2,
  function (x) sum(x * st$air_temperature) / sum(x)
)

print(predictions_multiplicative)

# ---- show-plot ----

plot(dh / 1000,
     h_distance,
     main = "Gaussian Kernel Values as a Function of Physical Distance",
     xlab = "Distance to Target Position (km)",
     ylab = "Kernel Value",
     cex = 0.4)
