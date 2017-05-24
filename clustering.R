library(mclust)
# cat("\014")
if (getwd() == '/home/neil'){
  setwd('projects/tsne-shots/')
}
# read csv and merge
data = read.csv('./data/tsne_shots.csv', header=TRUE)

# dekete non essential columns
ids = data['player_id']
data$player_id = NULL

seasons = data['season_id']
data$season_id = NULL

# clustering
BIC = mclustBIC(data)
mod1 = Mclust(data, x = BIC, G=1:20)
print('############################')
print('Overall Model')
print('############################')
summary_clusters = summary(mod1, parameters = TRUE)
distribution_clusters = summary(mod1, parameters = TRUE)$pro
mean_clusters = summary(mod1, parameters = TRUE)$mean
print(summary(mod1, parameters = TRUE))
data = cbind(data, player_id = ids)
data = cbind(data, season_id = seasons)
data = cbind(data, cluster = mod1$classification)

png('./plots/overall_cluster.png', width=1800,height=1300,res=300)
mod1dr = MclustDR(mod1, lambda = 1)
plot(mod1dr, what = "scatterplot", xaxt='n', yaxt='n')
title(main="Offense Evaluation Types", col.main="black")
dev.off()

# get cumalitive BIC score
print(BIC)
BIC <- rowSums(BIC)/14

png('./plots/overall_cluster_number.png', width=1800,height=1300,res=300)
plot(x=c(1:20),y=BIC,type="l",xlab="Number of Components", ylab="BIC Score")
axis(1, at=1:20)
title(main="BIC Score for Different Cluster Numbers", col.main="black")
dev.off()

write.csv(data,file='./data/overall_cluster.csv',row.names=FALSE,quote=FALSE)
