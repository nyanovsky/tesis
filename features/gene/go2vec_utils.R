library(wordspace)

prot_dist <- function(p1, p2, Y, emb){
  p1 <- as.character(p1)
  p2 <- as.character(p2)
  V1 <- Y[[p1]] # los GO ids asociados a la proteina p1
  V2 <- Y[[p2]]
  
  if(length(V1)==0 | length(V2)==0){
    return(NA)
  }
  
  M1 <- do.call(rbind, emb[unlist(V1)]) # a lo largo de cada fila esta el embedding de cada GO id
  M2 <- do.call(rbind, emb[unlist(V2)])
  
  
  if(is.null(M1) || is.null(M2)){
    return(NA)
  }
  
  M1 <- M1/rowNorms(M1)
  M2 <- M2/rowNorms(M2)
  
  D_12 <- M1 %*% t(M2) # en D_ij esta cos_sim(vi, vj)
  
  d_V1 <- mean(apply(D_12, 1, max)) 
  d_V2 <- mean(apply(D_12, 2, max))
  
  return(min(d_V1, d_V2))
}
v_prot_dist <- Vectorize(prot_dist, vectorize.args=c("p1", "p2"))

fast_neg_sampling <- function(dt, pos_samples){
  n = nrow(pos_samples)
  cols <- colnames(dt)
  ids = as.character(unique(c(dt[[cols[1]]], dt[[cols[2]]])))
  neg_samples <- data.frame(V2=numeric(n), V4=numeric(n))
  for(i in 1:nrow(neg_samples)){
    ps <- swap(sample(ids,2))
    while( nrow(dt[.(as.integer(ps[1]), as.integer(ps[2])), nomatch=0L])!=0 ){
      ps <- swap(sample(ids,2))
    }
    neg_samples[i,] <- c(ps[1], ps[2])
  }
  return(neg_samples)
}


clean_emb <- function(emb){
  col_name <- names(emb)
  emb <- setNames(as.list(emb[[col_name]]), row.names(emb))
  process_emb <- function(x){
    return (as.numeric(strsplit(x, " ")[[1]]))
  }
  emb <- lapply(emb, process_emb)
  nodos <- unname(unlist(lapply(emb, function(x) x[1])))
  emb <- setNames(emb, nodos)
  emb <- lapply(emb, function(x) x[2:length(x)])
  return(emb)
}

make_preds <- function(df_pos, df_neg, Y, emb){
  # df_pos y df_neg son los df con los enlaces pos y neg
  # Y es la lista que tiene prot:[terminos], emb los embeddings
  pred_int <- v_prot_dist(df_pos$V2, df_pos$V4, Y, emb)
  df_pos["mh_dist"] <- pred_int
  df_pos["label"] <- 1
  
  if(length(which(is.na(pred_int))) > 0){
    df_pos <- df_pos[-which(is.na(pred_int)), ]
  }
  
  pred_neg <- v_prot_dist(df_neg$V2, df_neg$V4, Y, emb)
  df_neg["mh_dist"] <- pred_neg
  df_neg["label"] <- 0
  
  if(length(which(is.na(pred_neg))) > 0){
    df_neg <- df_neg[-which(is.na(pred_neg)), ]
  }
  
  df_test <- rbind(df_pos, df_neg)
  return(df_test)
}



explore_hiperparams <- function(param_grid, df_pos, df_neg, Y){
  library(doMC)
  library(foreach)
  library(progressr)
  registerDoMC(cores = 40)
  progress <- progressor(along = 1:nrow(param_grid))
  
  results_df <- foreach(i = 1:nrow(param_grid), .combine = rbind) %dopar%{
    params <- param_grid[i, ]
    l <- params$l
    d <- params$d
    k <- params$k
    r <- params$r
    
    progress()
    
    
    system(glue("./node2vec -i:goa_mf.txt -o:emb_{d}_{l}_{k}_{r} -d:{d} -l:{l} -k:{k} -r:{r} "))
    emb <- read.table(glue("emb_{d}_{l}_{k}_{r}"), sep='\t', header=TRUE)
    emb <- clean_emb(emb)
    df_test <- make_preds(df_pos, df_neg, Y, emb)
    
    auc <- auc(roc(df_test$label, df_test$mh_dist))
    
    data.frame(l = l,
               d = d,
               k = k,
               r = r,
               auc = auc)
  }
  
  return(results_df)
}

swap <- function(row){
  if(row[2] < row[1]){
    return(c(row[2], row[1]))
  }
  else{
    return(row)
  }
}


