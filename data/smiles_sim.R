library(rcdk)
library(fingerprint)
library(readr)
library(data.table)
options(bitmapType = 'cairo')

lines <- readLines("~/data/ChCh/cid_to_smiles.txt")
data <- strsplit(lines, "\t")

not_mapped <- which(unlist(lapply(data, function(x) length(x) <2)))
data <- data[-not_mapped]

df_cid_to_smile <- do.call(rbind, lapply(data, function(x) as.data.frame(t(x), stringsAsFactors = FALSE)))

colnames(df_cid_to_smile) <- c("PubChem_CID", "SMILES")

smiles <- df_cid_to_smile$SMILES
cids <- df_cid_to_smile$PubChem_CID

### PARSE SMILES

mols <- parse.smiles(smiles)

print(sum(unlist(lapply(mols, is.null))))

# No hay nulls

fps <- lapply(mols, function(x) get.fingerprint(x, type="circular"))

library(doParallel)
library(doMC)
library(data.table)

registerDoMC(40)

cutoff <- 0.5

sims <- c()

### CALCULATE TANIMOTO SIM
results <- foreach(i = 1:(length(fps) - 1), .combine = rbind, .packages = 'data.table') %dopar% {
  temp_results <- NULL
  for (j in (i + 1):length(fps)) {
    sim <- distance(fps[[i]], fps[[j]], method="tanimoto")
    if (sim > cutoff) {
      temp_results <- rbind(temp_results, data.table(CID1 = cids[i], CID2 = cids[j], Similarity = sim))
    }
  }
  return(temp_results)
}

sims <- foreach(i = 1:(length(fps) - 1), .combine = c, .packages = 'data.table') %dopar% {
  temp_sims <- numeric()
  for (j in (i + 1):length(fps)) {
    sim <- distance(fps[[i]], fps[[j]], method="tanimoto")
    temp_sims <- c(temp_sims, sim)
  }
  return(temp_sims)
}

plot(density(sims))

fwrite(results, "~/data/ChCh/tani_net_05.csv")
