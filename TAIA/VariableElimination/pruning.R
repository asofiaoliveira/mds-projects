pruneNetwork <- function(net, query, evidence = NULL, debug = FALSE) {
  # Prunes the tree according to query and evidence. This pruning will not impact the result of VE
  #
  # Args:
  #   net: network (list retrieved by read.bif()).
  #   query: string representing the query variable.
  #   evidence: array representing the evidence variables.
  #   debug: debug flag
  #
  # Returns:
  #   A list of relevant nodes of the tree.
  # Notes:
  #   Not very efficient.
  if (debug) cat("\n\n --- PRUNING DEBUG --- \n")
    
  relevant_nodes = rep(FALSE, length(net))
  names(relevant_nodes) = names(net)
  
  # Query nodes are relevant
  relevant_nodes[[query]] = TRUE
  
  # If a node is relevant, then its parents are relevant too
  parents = net[[query]]$parents
  while(length(parents) != 0){
    relevant_nodes[[parents[1]]] = TRUE
    parents = c(parents, net[[parents[1]]]$parents)
    parents = parents[-1]
  }

  if (debug) cat("Relevant nodes before evidence:", names(relevant_nodes[relevant_nodes==TRUE]), "\n")
  
  
  # Any evidence node is relevant if its a descendant of a relevant node
  relevant_evidence = rep(FALSE, length(evidence))
  names(relevant_evidence) = evidence
  if(length(evidence) > 0){
    
    # verify if evidence is already relevant
    temp = relevant_evidence[names(relevant_nodes[relevant_nodes])]
    temp = names(temp[!is.na(temp)])
    relevant_evidence[temp] = TRUE
    
    for(i in 1:length(relevant_nodes)){ 
      if(all(relevant_evidence)) break # if every evidence is already relevant, there's no point in cointinuing the cycle
      if(relevant_nodes[[i]] == TRUE){
        # derive descendants of this relevant node
        descendants = c(net[[names(relevant_nodes)[i]]], net[[names(relevant_nodes)[i]]]$children)
        children = net[[names(relevant_nodes)[i]]]$children
        while(length(children) != 0){
          children = c(children, net[[children[1]]]$children)
          descendants = c(descendants, net[[children[1]]]$children)
          children = children[-1]
        }
        
        # check if any evidence is a descendant of this node, and if so, the evidence is also relevant
        for(j in 1:length(evidence)){
          if(evidence[j] %in% descendants)
            relevant_evidence[[evidence[j]]] = TRUE
        }
      }
    }
  }
  
  relevant_evidence = names(relevant_evidence[relevant_evidence==TRUE])
  if (debug) cat("Relevant evidence:", relevant_evidence, "\n")
  
  # If a node Z is relevant, then its parents are relevant too
  # Done for the 2nd time because the evidence might bring new relevant nodes
  if(length(relevant_evidence) > 0){
    for(i in 1:length(relevant_evidence)){
      parents = net[[relevant_evidence[i]]]$parents
      while(length(parents) != 0){
        relevant_nodes[[parents[1]]] = TRUE
        parents = c(parents, net[[parents[1]]]$parents)
        parents = parents[-1]
      }
    }
  }
  
  # Finally, combine relevant nodes with relevant evidence
  relevant_nodes = names(relevant_nodes[relevant_nodes==TRUE])
  relevant_nodes = c(relevant_nodes, relevant_evidence) %>% unique()
  
  if (debug) cat("Relevant nodes after evidence:", relevant_nodes, "\n")
  if (debug) cat(" --- END PRUNING DEBUG --- \n\n")
    
  return(relevant_nodes)
}
