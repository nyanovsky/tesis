library("org.Hs.eg.db")
library("AnnotationDbi")
library("dplyr")
ChG_df <- read_csv("~/data/ChG/ChG_df.csv")
GeneIds <- as.character(ChG_df$`GeneID (NCBI)`)

pfam_annotations_df <- AnnotationDbi::select(org.Hs.eg.db, keys=GeneIds, columns="PFAM", keytype="ENTREZID")

not_annotated <- which(is.na(pfam_annotations_df$PFAM))
pfam_annotations_df <- pfam_annotations_df[-not_annotated,]

pfam_annotations_list <- split(pfam_annotations_df$ENTREZID, pfam_annotations_df$PFAM)

pfam_annotations_list <- lapply(pfam_annotations_list, unique)

total_annotations <- sum(unlist(lapply(pfam_annotations_list, function(x) length(x))))

IC <- unlist(lapply(pfam_annotations_list, function(x) -log2(length(x)/total_annotations)))

no_informativos <- which(IC < -log2(20/total_annotations))

pfam_annotations_list <- pfam_annotations_list[-no_informativos]

pfam_annotations_list <- lapply(pfam_annotations_list, sort)

gene_edges <- list()

for (pfam_id_anot in pfam_annotations_list){
  if (length(pfam_id_anot) >1){
    gene_edges <- c(gene_edges, combn(pfam_id_anot, 2, simplify = FALSE))
  }
}

gene_edges_df <- do.call(rbind, lapply(gene_edges, function(x) data.frame(src=x[1], trgt=x[2])))

gene_edges_df <- distinct(gene_edges_df)

fwrite(gene_edges_df, "~/data/GG/pfam_proy_net.csv")
