library(dplyr)

pfam_sim_df <- read.csv("~/data/GG/pfam_w_proy_net.csv")
# mf_anot_list <-  split(mf_go_df$V2, mf_go_df$V1)
mf_anot_list <- split(informative_annotations$V2, informative_annotations$V1)

anotadas_src <- sapply(as.character(pfam_sim_df$src), function(x) x %in% names(mf_anot_list))
anotadas_trgt <- sapply(as.character(pfam_sim_df$trgt), function(x) x %in% names(mf_anot_list))
interacciones_anotadas <- anotadas_src&anotadas_trgt 

pfam_sim_df <- filter(pfam_sim_df, interacciones_anotadas)
sim_df_sorted <- as.data.frame(t(apply(pfam_sim_df, 1, swap)))
sim_dt_sorted <-  data.table(sim_df_sorted)
setkey(sim_dt_sorted, V1, V2)


pfam_sim_df <- as.data.frame(pfam_sim_df)
colnames(pfam_sim_df) <- c("V2", "V4")
pfam_sim_df$V2 <- as.character(pfam_sim_df$V2)
pfam_sim_df$V4 <- as.character(pfam_sim_df$V4)
prot_ids <- unique(unlist(pfam_sim_df))

#### Test ####
neg_samples <- fast_neg_sampling(sim_dt_sorted, pfam_sim_df)
system("./node2vec -i:informative_goa_mf.txt -o:test_embs -d:32 -l:40 -k:6 -r:5")
test_embs <- read.table("test_embs", sep='\t', header=T)
clean_test_embs <- clean_emb(test_embs)
test_preds <- make_preds(pfam_sim_df, neg_samples, mf_anot_list, clean_test_embs) # 0.7 AUC

### Full hyperparam exploration ###
param_grid <- expand.grid(l = c(20,40,80,100),
                                        d = c(8,16,32,64),
                                        k = c(3,6,9,12),
                                        r = c(5,8,10)
)

results_df <- explore_hiperparams(param_grid, pfam_sim_df, neg_samples, mf_anot_list)
results_df[which.max(results_df$auc), ] # l:20 d:64 k:3 r:8

best_embs <- read.table("emb_64_20_3_8", sep="\t", header=T)
best_embs <- clean_emb(best_embs)

prots_in_graph <- unique(informative_annotations$V1)
best_prot_embs <- best_embs[names(best_embs) %in% prots_in_graph]

lapply(best_prot_embs, write, "prot_features_64.txt", append=T, ncolumns=20000)
write(names(best_prot_embs), "prot_features_ids.txt")
