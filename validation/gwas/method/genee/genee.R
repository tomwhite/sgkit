#!/usr/bin/env Rscript

library("genee")

load("data/Simulated_Data_Example/Simulated_LD.RData")
load("data/Simulated_Data_Example/Simulated_Summary_Statistics.RData")
load("data/Simulated_Data_Example/gene_list.RData")

# use alpha = -1 for OLS
result = genee(mydata, ld, alpha = -1, gene_list = gene_list)

write.csv(mydata, "data/Simulated_Data_Example/mydata.csv")
write.csv(ld, "data/Simulated_Data_Example/ld.csv")
write.csv(result, "data/Simulated_Data_Example/result.csv")