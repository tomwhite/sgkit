#!/usr/bin/env Rscript

library("genee")

# Simulated data

load("data/Simulated_Data_Example/Simulated_LD.RData")
load("data/Simulated_Data_Example/Simulated_Summary_Statistics.RData")
load("data/Simulated_Data_Example/gene_list.RData")

# use alpha = -1 for OLS
result = genee(mydata, ld, alpha = -1, gene_list = gene_list)

write.csv(mydata, "data/Simulated_Data_Example/mydata.csv")
write.csv(ld, "data/Simulated_Data_Example/ld.csv")
write.csv(result, "data/Simulated_Data_Example/result.csv")

# Real data

load("data/Real_Data_Example/Real_LD.RData")
load("data/Real_Data_Example/Real_Summary_Statistics.RData")
load("data/glist.hg19.rda")

# TODO: run genee

write.csv(mydata, "data/Real_Data_Example/mydata.csv")
write.csv(ld, "data/Real_Data_Example/ld.csv")
write.csv(glist.hg19, "data/glist.hg19.csv")

# Write gene list to text file
all_chr=as.numeric(mydata[,1])
all_pos=as.numeric(mydata[,3])
temp = genee_list(glist.hg19, all_chr, all_pos, 50000, 50000)
gene_info = temp[[1]]
gene_list = temp[[2]]
for (i in 1:length(gene_list)) {
  cat(paste(gene_list[[i]], sep="' '", collapse=","), file="data/Real_Data_Example/gene_list.txt", sep="\n", append=TRUE)
}