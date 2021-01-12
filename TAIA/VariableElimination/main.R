#
# Verify installation of required packages
#
packages <- c("bnlearn", "dplyr", "purrr", "collections")
installed <- installed.packages()

for (package in packages) {
    if ( !(package %in% rownames(installed)) )
       stop(sprintf("Package %s required!", package))
    
    suppressMessages(library(package, character.only=TRUE))
}

rm(packages, installed)

#
# Load required source files
#
sources <- c("selection", "utils", "pruning", "algorithm")

for (source in sources) {
    file <- sprintf("%s.R", source)

    if (!file.exists(file))
        stop(sprintf("File %s required!", file))
    
    source(file)
}

rm(sources)

#
# Were cmd-args provided?
#
args <- commandArgs(trailingOnly = TRUE)

prune <- FALSE
debug <- FALSE

if (length(args) > 0) {
    for (arg in args) {
        if (as.character(arg) == "--debug")
            debug <- TRUE
        if (as.character(arg) == "--prune")
            prune <- TRUE
    }
}

#
# Network to be used?
#
cat("Network(.bif)? ")
net <- readLines("stdin", n=1)

if (!file.exists(net))
   stop(sprintf("File %s doesn't exist!", net))

net <- read.bif(net)

#
# Program I/O loop
#
while (TRUE) {
    # query + evidence input
    op <- inputOperation()

    if (!validateQuery(op$query, net) ||
        !validateEvidence(op$evidence, net)) {
        next
    }
    
    if (debug) cat("Query:", op$query, "\nEvidence:", op$evidence, fill = TRUE)
    
    # prune network, if opted in
    if(prune) {
      net1 <- net[pruneNetwork(net, op$query, op$evidence, debug)]
      cat("Pruned", length(net)-length(net1), "nodes out of", length(net), "total nodes\n")
    }
    else net1 <- net

    # elimination order input + generation
    order <- NULL
    if (!all(names(net1) %in% c(op$query, op$evidence))) {
      heur <- orderOptions[ menutize(orderOptions, "Choose elimination order:") ]
      order <- greedyOrdering(net1, op$query, op$evidence, heur, debug)
      
      cat(sprintf("Elimination Order: %s", net1[[order[[1]]]]$node))
      if (length(order) > 1){
        for (idx in 2:length(order))
          cat(sprintf(",%s", net1[[order[[idx]]]]$node))
      }
      cat("\n")
    }
    
    # call variable elimination
    op$evidence = op$evidence[which(op$evidence %in% names(net1))]
    result = ve(net1, op$query, op$evidence, order, debug)
    print(result)
    
    # exit ?
    cat("Exit? (y/N) ")
    ans <- readLines("stdin", n=1)

    if (ans == "y")
        break
}

cat("Bye...", fill = TRUE)
