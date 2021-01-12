makeAppears = function(nodes, var){
  # Determine where a variable appears in the nodes
  # 
  # Args:
  #   nodes: conditional probability tables
  #   var: variable to search
  #
  # Returns:
  #   An array with the indexes in which the variable var appears in the list of nodes
  appears = c()
  for(i in 1:length(nodes)){
    if(var %in% names(nodes[[i]]))
      appears = c(appears, i)
  }
  return(appears)
}

ve = function(net, query, evidence, order, debug = FALSE){
  # Variable elimination algorithm
  # 
  # Args:
  #   net: bayesian network in which to perform VE
  #   query: variable of interest
  #   evidence: variables with apriori fixed values
  #   order: order of variable elimination
  #   debug: debug flag
  #
  # Returns:
  #   A table with the variable "query" probabilities conditioned by the given evidence
  if (debug) cat("\n\n --- VE DEBUG --- \n")

  # if a variable has no parents and there is no evidence, then its probability is given directly by the probability table
  if(is_empty(net[[query]]$parents) & is.na(evidence[1])) return(as.vector(net[[query]]$prob))
  
  # Simplifying "net" into a list "nodes" (only with the respective cond. prob. table)
  nodes = list()
  
  for(i in 1:length(net)){
    # create nodes
    nodes[[i]] = as.data.frame(net[[i]]$prob)
    if(is_empty(net[[i]]$parents))
      # if a node doesn't have parents, then the probability table won't have the node's name
      # so if that's the case, we change the dataframe's names to the node and "Freq"
      names(nodes[[i]]) = c(net[[i]]$node, "Freq")
  }
  
  names(nodes) = names(net)
  
  # Conditioning data given evidence
  if(!is.na(evidence[1])){
    for(i in 1:length(evidence)){
      levels = levels(nodes[[evidence[i]]][evidence[i]][,])
      value = levels[1]
      # ask the user what level was observed
      value <- levels[ menutize(levels, sprintf("Choose a level for the evidence variable %s", evidence[i])) ]
      for(j in 1:length(nodes)){
        if(evidence[i] %in% names(nodes[[j]]))
          nodes[[j]] = nodes[[j]][which(nodes[[j]][evidence[i]] == value),]
      }
    }
  }
  
  if(debug) print(nodes)
    
  for(i in order){
    var = net[[i]]$node
    
    if(debug) cat("Next variable to eliminate:", i, "-", var, "\n")
    
    appears = makeAppears(nodes, var)
    
    # get nodes in which variable var appears
    n = nodes[appears]
    
    if(debug){
      cat("Nodes in which to eliminate variable:", var, "\n")
      print(n)
    }
    
    # if variable var is referenced in more than 1 table, join tables
    if(length(n)>1){
      for(j in 2:length(n)){
        # changing names of "Freq." columns so that those columns won't join with each other
        names(n[[j]])[length(n[[j]])] = paste("Freq", j, sep = '.')
      }
    
      n = n %>% reduce(merge)
      if(debug) print(n)
      
      # multiply frequncies
      n$Freq = reduce(select(n, starts_with("Freq")),`*`)
      
      # remove columns with all frequencies except the one calculated just before
      n = n %>% select(-starts_with("Freq."))
    }
    
    # controlling the type of n
    if(length(n) == 1){
      n = n[[1]]
    }
    
    # sum in variable var values
    n = n %>% group_by(.dots=setdiff(names(n), c(var, "Freq"))) %>% summarise(Freq = sum(Freq)) %>% data.frame() # is this correct?
    if(debug){
      cat("New node to add:\n")
      print(n)
    }
      
    # delete used nodes
    nodes[appears] = NULL
    
    # sometimes there might be a node that the only column is the frequency
    if(length(names(n))>1){
      nodes = append(nodes, list(n))
    }
    
    if(debug){
      cat("Updated nodes:\n")
      print(nodes)
    }
  }
  
  # Join evidence variables
  if(length(nodes)>1){
    for(j in 2:length(nodes)){
      # changing names of "Freq." columns so that those columns won't join with each other
      names(nodes[[j]])[length(nodes[[j]])] = paste("Freq", j, sep = '.')
    }
    
    nodes = nodes %>% reduce(merge)
    
    # multiply frequncies
    nodes$Freq = reduce(select(nodes, starts_with("Freq")),`*`)
    
    # remove columns with all frequencies except the one calculated just before
    nodes = nodes %>% select(-starts_with("Freq."))
  }
  
  if(length(nodes) == 1){
    nodes = nodes[[1]]
  }
  
  if(debug){
    cat("Final non-normalized probability table:\n")
    print(nodes)
  }
  
  # compute denominator -- sum in the query variable
  den = nodes %>% summarise(den = sum(Freq)) %>% pull(den)
  
  if(debug) cat("Dividing frequencies by", den, "to normalize the probability.\n")
  
  # divide by denominator
  result = nodes %>% mutate(Freq, Freq = Freq/den) %>% pull(Freq)

  if (debug) cat(" --- END VE DEBUG --- \n\n")
    
  return(result)
}
