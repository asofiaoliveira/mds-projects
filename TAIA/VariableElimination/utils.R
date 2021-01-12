menutize <- function(options, title) {
    # Behaves the same as R's menu() function,
    # only I/O operations are changed so we can
    # use this function in R scripts (Rscript is non-interactive).
    #
    # Args:
    #   options: Character array with options to be listed.
    #   title: Menu's title.
    #
    # Returns:
    #   Index of the selected option.
    cat(title, fill=TRUE);
    cat("\n");

    for (op in 1:length(options))
        cat(sprintf("%d: %s", op, options[op]), fill=TRUE)

    cat("\nSelection: ")
    
    exit <- FALSE
    selec <- 0
    
    while (!exit) {
        inp <- readLines("stdin", n=1)

        if (is.na(inp) || inp == "") {
            cat("Enter an item from the menu\nSelection: ")
        } else {
            inp <- suppressWarnings(as.numeric(inp))
            
            if (is.na(inp) || inp <= 0 || inp > length(options)) {
                cat("Invalid selection\nSelection: ")
            } else {
                selec <- inp
                exit <- TRUE
            }
        }
    }

    return(selec)
}


inputOperation <- function() {
    # Asks the user for query + evidence variables 
    # and handles its format.
    #
    # Returns:
    #   list$query - query variable (as string).
    #   list$evidence - evidence variables (as array).
    cat("Pr? ")
    input <- readLines("stdin", n=1)

    split <- strsplit(input, split = "|", fixed = TRUE)[[1]]

    query <- gsub(" ", "", split[1])
    evidence <- gsub(" ", "", split[2])

    if (!is.na(evidence))
        evidence <- strsplit(evidence, split = ",", fixe = TRUE)[[1]]

    return( list("query" = query, "evidence" = evidence) )
}

validateQuery <- function(query, net) {
    # Checks if the query-var is valid in the network being used.
    #
    # Args:
    #   query: string representing the query variable.
    #   net: network (list retrieved by read.bif()).
    #
    # Returns:
    #   TRUE if valid query-var, FALSE otherwise.
    if (is.null(net[[query]])) {
        cat(sprintf("Query-var %s doesn't exist in the network!", query), fill = TRUE)
        return(FALSE)
    }

    return(TRUE)
}

validateEvidence <- function(evidence, net) {
    # Checks if the evidence-vars are valid in the network being used.
    #
    # Args:
    #   evidence: array representing the evidence variables.
    #   net: network (list retrieved by read.bif()).
    #
    # Returns:
    #   TRUE if valid evidence-vars, FALSE otherwise.
    if (is.na(evidence))
        return(TRUE)

    for (var in evidence) {
        if (is.null(net[[var]])) {
            cat(sprintf("Evidence-var %s doesn't exist in the network!", var), fill = TRUE)
            return(FALSE)
        }
    }

    return(TRUE)
}
