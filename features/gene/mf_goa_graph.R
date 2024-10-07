library("GO.db")
library("org.Hs.eg.db")
library("dplyr")
library("igraph")


ChG_df <- read_csv("~/data/ChG/ChG_df.csv")
GeneIds <- as.character(ChG_df$`GeneID (NCBI)`)
GeneIds <- unique(GeneIds)

mf_children_df <- as.data.frame(GOMFCHILDREN)
mf_children_df <- go_children_df[,-3]
colnames(mf_children_df) <- c("V1", "V2")

prot_go_df <- as.data.frame(org.Hs.egGO2EG)
mf_go_df <- prot_go_df[prot_go_df$Ontology=="MF",]

ev_codes <- c("EXP", "IDA", "IPI", "IMP","IGI","IEP")
mf_go_df <- filter(mf_go_df, Evidence %in% ev_codes)
mf_go_df <- distinct(mf_go_df[,-c(3,4)])

mf_go_df$go_id <- unlist(lapply(mf_go_df$go_id, function(x) paste(1,substring(x,first=4), sep = "")))
mf_children_df$V1 <- unlist(lapply(mf_children_df$V1, function(x) paste(1,substring(x,first=4), sep = "")))
mf_children_df$V2 <- unlist(lapply(mf_children_df$V2, function(x) paste(1,substring(x,first=4), sep = "")))
# transformo los go_ids de "GO:..." a "1..." por un tema del input de node2vec


colnames(mf_go_df) <- c("V1", "V2")
df_goa_mf <- rbind(mf_children_df, mf_go_df)
df_goa_mf <- apply(df_goa_mf, 2, as.numeric)

mf_children_df <- as.data.frame(mf_children_df)
mf_go_df <- as.data.frame(mf_go_df)

write.table(df_goa_mf, "~/data/GG/mf_goa/goa_mf.txt", row.names = F, col.names = F)
write.table(mf_children_df, "~/data/GG/mf_goa/mf_terms_edgelist.csv", row.names = F)
write.table(mf_go_df, "~/data/GG/mf_goa/mf_anotations_edgelist.csv", row.names = F)

##### Sin terminos poco informativos #####

go2all_df <- as.data.frame(org.Hs.egGO2ALLEGS)
go2all_mf <- go2all_df[go2all_df$Ontology=="MF",]
go2all_mf <- filter(go2all_mf, Evidence %in% ev_codes)
go2all_mf <- go2all_mf[,-c(3,4)]
go2all_mf$go_id <- unlist(lapply(go2all_mf$go_id, function(x) paste(1,substring(x,first=4), sep = "")))

go2all_list <- split(go2all_mf$gene_id, go2all_mf$go_id)

tot_prots <- length(unique(go2all_mf$gene_id))
  
IC_calc <- function(goID){
  id <- as.character(goID)
  return(-log2(length(go2all_list[[id]])/tot_prots))
  }
v_IC_calc <- Vectorize(IC_calc)

goIds <- unique(go2all_mf$go_id)
IC_df <- data.frame(go_id = goIds, IC=rep(0, length(goIds)))
IC_df$IC <- v_IC_calc(goIds)
plot(density(IC_df$IC))

less_informative <- IC_df[IC_df$IC < -log2(0.05), "go_id"] 
more_informative <- IC_df[IC_df$IC >= -log2(0.05), "go_id"]

informative_annotations <- filter(mf_go_df, V2 %in% more_informative)
filtered_goa_mf <- rbind(mf_children_df, informative_annotations)
filtered_goa_mf <- apply(filtered_goa_mf, 2, as.numeric)
write.table(filtered_goa_mf, "informative_goa_mf.txt", row.names = F, col.names = F)
