library("org.Hs.eg.db")
library("AnnotationDbi")
library("dplyr")
library("igraph")
ChG_df <- read_csv("~/data/ChG/ChG_df.csv")
GeneIds <- as.character(ChG_df$`GeneID (NCBI)`)
GeneIds <- unique(GeneIds)


pfam_annotations_df <- AnnotationDbi::select(org.Hs.eg.db, keys=GeneIds, columns="PFAM", keytype="ENTREZID")
not_annotated <- which(is.na(pfam_annotations_df$PFAM))
pfam_annotations_df <- pfam_annotations_df[-not_annotated,]

gene_nodes <- unique(pfam_annotations_df$ENTREZID)

g <- graph_from_data_frame(pfam_annotations_df, directed = FALSE)
V(g)$type <-  V(g)$name %in% gene_nodes

inc_matrix <- as_incidence_matrix(g)

pfam_degs <- rowSums(inc_matrix)
gene_degs <- colSums(inc_matrix)

D_pfam <- diag(1/pfam_degs)

gene_proy <- t(inc_matrix) %*% D_pfam %*% inc_matrix
diag(gene_proy) <- 0
weights <- gene_proy[lower.tri(gene_proy)]
plot(density(log10(weights[weights>0])))

cutoff <- 10^(-1.5)
gene_proy[] <- vapply(gene_proy, function(x) x>cutoff, numeric(1))

gene_proy_graph <- graph_from_adjacency_matrix(gene_proy, mode="undirected")
gene_proy_edge_list <- as.data.frame(get.edgelist(gene_proy_graph))
colnames(gene_proy_edge_list) <- c("src", "trgt")
fwrite(gene_proy_edge_list, "~/data/GG/pfam_w_proy_net.csv")
