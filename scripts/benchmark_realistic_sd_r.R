args <- commandArgs(trailingOnly = TRUE)

option_value <- function(name, default) {
  prefix <- paste0("--", name, "=")
  matched <- args[startsWith(args, prefix)]
  if (length(matched) == 0) {
    return(default)
  }
  sub(prefix, "", matched[[1]], fixed = TRUE)
}

time_call <- function(fun, repeats) {
  times <- replicate(repeats, system.time(invisible(fun()))[["elapsed"]])
  c(mean = mean(times), min = min(times), max = max(times))
}

fixture <- option_value(
  "fixture",
  "tests/fixtures/finance/sp500_daily_returns_2019_2023.csv"
)
repeats <- as.integer(option_value("repeats", "3"))
max_repeats <- as.integer(option_value("max-repeats", "1"))
output <- option_value("output", "")

returns <- read.csv(fixture, check.names = FALSE)
max_columns <- ncol(returns) - 1

cases <- data.frame(
  function_name = c(
    "sd_efficient_set",
    "nns_sd_cluster",
    "sd_efficient_set",
    "nns_sd_cluster",
    "sd_efficient_set",
    "nns_sd_cluster",
    "sd_efficient_set",
    "nns_sd_cluster",
    "sd_efficient_set",
    "nns_sd_cluster",
    "sd_efficient_set",
    "nns_sd_cluster",
    "sd_efficient_set",
    "nns_sd_cluster",
    "sd_efficient_set",
    "nns_sd_cluster",
    "sd_efficient_set",
    "nns_sd_cluster"
  ),
  rows = c(
    252, 252,
    252, 252,
    252, 252,
    252, 252,
    252, 252,
    252, 252,
    1257, 1257,
    1257, 1257,
    1257, 1257
  ),
  columns = c(
    50, 50,
    100, 100,
    250, 250,
    max_columns, max_columns,
    50, 50,
    100, 100,
    100, 100,
    250, 250,
    max_columns, max_columns
  ),
  degree = c(
    1, 1,
    1, 1,
    2, 2,
    2, 2,
    2, 2,
    2, 2,
    2, 2,
    2, 2,
    2, 2
  )
)

library(NNS)

results <- data.frame(
  function_name = character(),
  rows = integer(),
  columns = integer(),
  degree = integer(),
  repeats = integer(),
  mean_seconds = numeric(),
  min_seconds = numeric(),
  max_seconds = numeric(),
  result_size = integer()
)

for (index in seq_len(nrow(cases))) {
  function_name <- cases$function_name[[index]]
  rows <- cases$rows[[index]]
  columns <- cases$columns[[index]]
  degree <- cases$degree[[index]]
  case_repeats <- if (columns == max_columns && rows == nrow(returns)) max_repeats else repeats
  mat <- as.matrix(returns[seq_len(rows), 2:(columns + 1)])

  if (function_name == "sd_efficient_set") {
    result <- NNS::NNS.SD.efficient.set(
      mat,
      degree = degree,
      type = "discrete",
      status = FALSE
    )
    result_size <- length(result)
    timed <- time_call(function() {
      NNS::NNS.SD.efficient.set(
        mat,
        degree = degree,
        type = "discrete",
        status = FALSE
      )
    }, case_repeats)
  } else {
    result <- NNS::NNS.SD.cluster(
      mat,
      degree = degree,
      type = "discrete",
      min_cluster = 1,
      dendrogram = FALSE
    )
    result_size <- length(unlist(result$Clusters, use.names = FALSE))
    timed <- time_call(function() {
      NNS::NNS.SD.cluster(
        mat,
        degree = degree,
        type = "discrete",
        min_cluster = 1,
        dendrogram = FALSE
      )
    }, case_repeats)
  }

  results <- rbind(results, data.frame(
    function_name = function_name,
    rows = rows,
    columns = columns,
    degree = degree,
    repeats = case_repeats,
    mean_seconds = timed[["mean"]],
    min_seconds = timed[["min"]],
    max_seconds = timed[["max"]],
    result_size = result_size
  ))
}

if (output != "") {
  write.csv(results, output, row.names = FALSE, quote = FALSE)
} else {
  write.csv(results, stdout(), row.names = FALSE, quote = FALSE)
}
