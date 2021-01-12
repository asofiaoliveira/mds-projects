orderOptions <- c("min-table", "min-children", "min-parents", "min-neighbors", "min-weight")

greedyOrdering <- function(x, query, observed, heur = orderOptions, debug = FALSE) {
    # Retrieve elimination order based on selected heuristic.
    # 
    # Args:
    #   x: Network (list retrieved by read.bif()).
    #   query: Query variable (node in the net).
    #   observed: Evidence variable/s (nodes in the net).
    #   heur: Selection heuristic (defaults to min-table).
    #   debug: debug flag
    #
    # Returns:
    #   An array ordered by node (index) elimination order.
    if (debug) cat("\n\n --- GREEDY ORDERING DEBUG --- \n")
    
    heur <- sub("-", "", match.arg(heur))

    pq <- PriorityQueue$new()
    
    for (i in 1:length(x)) {
        node <- x[[i]]$node

        # ignore current node if query or part of evidence
        if (node == query || node %in% observed) {
            if (debug) cat("Ignoring", node, fill=TRUE)
            next
        }
        
        # enqueue current node's heuristic value
        val <- do.call(heur, list(x,i))
        pq$push(i, -val) # -val because PriorityQueue behaves as max-heap (we want min-heap)
        if (debug) cat(sprintf("Enqueuing %s with value %d\n", node, val))
    }

    if (debug) cat(" --- END GREEDY ORDERING DEBUG --- \n\n")

    return( unlist(pq$as_list()) )
}

mintable <- function(net, node) {
    # Computes min-table heuristic value for given node
    #
    # Args:
    #   net: Network (list retrieved by read.bif()).
    #   node: Node to compute value for.
    #
    # Returns:
    #   Heuristic value for given node.
    return ( length(net[[node]]$prob) )
}
    
minchildren <- function(net, node) {
    # Computes min-children heuristic value for given node
    #
    # Args:
    #   net: Network (list retrieved by read.bif()).
    #   node: Node to compute value for.
    #
    # Returns:
    #   Heuristic value for given node.
    return ( length(net[[node]]$children) )
}

minparents <- function(net, node) {
    # Computes min-parents heuristic value for given node
    #
    # Args:
    #   net: Network (list retrieved by read.bif()).
    #   node: Node to compute value for.
    #
    # Returns:
    #   Heuristic value for given node.
    return ( length(net[[node]]$parents) )
}

minneighbors <- function(net, node) {
    # Computes min-neighbors heuristic value for given node
    #
    # Args:
    #   net: Network (list retrieved by read.bif()).
    #   node: Node to compute value for.
    #
    # Returns:
    #   Heuristic value for given node.
    return (minchildren(net, node) + minparents(net, node))
}

minweight <- function(net, node) {
    # Computes min-weight heuristic value for given node
    #
    # Args:
    #   net: Network (list retrieved by read.bif()).
    #   node: Node to compute value for.
    #
    # Returns:
    #   Heuristic value for given node.
    weight <- 1
    node <- net[[node]]
    
    for (parent in node$parents) {
        weight <- weight * length(net[[parent]]$prob)
    }

    for (child in node$children) {
        weight <- weight * length(net[[child]]$prob)
    }

    return (weight)
}
