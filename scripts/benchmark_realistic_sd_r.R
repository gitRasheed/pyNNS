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
  mean(times)
}

fixture <- option_value(
  "fixture",
  "tests/fixtures/finance/sp500_daily_returns_2019_2023.csv"
)
repeats <- as.integer(option_value("repeats", "3"))

returns <- read.csv(fixture, check.names = FALSE)

cases <- data.frame(
  rows = c(252, 252, 252, 252, 252, 1257),
  columns = c(50, 50, 100, 100, 250, 100),
  degree = c(1, 2, 1, 2, 2, 2)
)

library(NNS)

cat("function,rows,columns,degree,mean_seconds\n")

for (index in seq_len(nrow(cases))) {
  rows <- cases$rows[[index]]
  columns <- cases$columns[[index]]
  degree <- cases$degree[[index]]
  mat <- as.matrix(returns[seq_len(rows), 2:(columns + 1)])

  invisible(NNS::NNS.SD.efficient.set(
    mat,
    degree = degree,
    type = "discrete",
    status = FALSE
  ))
  efficient_seconds <- time_call(function() {
    NNS::NNS.SD.efficient.set(
      mat,
      degree = degree,
      type = "discrete",
      status = FALSE
    )
  }, repeats)
  cat(sprintf(
    "sd_efficient_set,%d,%d,%d,%0.8f\n",
    rows,
    columns,
    degree,
    efficient_seconds
  ))

  invisible(NNS::NNS.SD.cluster(
    mat,
    degree = degree,
    type = "discrete",
    min_cluster = 1,
    dendrogram = FALSE
  ))
  cluster_seconds <- time_call(function() {
    NNS::NNS.SD.cluster(
      mat,
      degree = degree,
      type = "discrete",
      min_cluster = 1,
      dendrogram = FALSE
    )
  }, repeats)
  cat(sprintf(
    "nns_sd_cluster,%d,%d,%d,%0.8f\n",
    rows,
    columns,
    degree,
    cluster_seconds
  ))
}
