# FashionMNIST with decision trees and transformers
Comparing different decision tree algorithms for the FashionMNIST classification problem

I compare:

- Regular decision trees, accounting for seed dependency
- Random forests
- Scikit's bagging algorithm
- XGboost
- SimpleVIT Transformer!

Contrary to popular belief, trees do give a pretty good image classification accuracy that rivals CNNs, up to 90%, and with essentially no data preprocessing or normalization. 

SimpleViT is slow, and I didn't manage to get it past 90% accuracy.
