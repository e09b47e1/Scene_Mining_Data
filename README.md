# Datasets and Codes of Scene Mining Task (for reviewers)  #

These are anonymous open source datasets and codes for scene mining task.
You can find a more concrete description about the datasets in Section Experiment of our submitted paper.
In order to comply with the submission policy of the conference, there is no description of content except file format here.
The datasets will be released formally in camera ready submission. 

## Datasets Introduction  ##
We construct four different datasets based on four major category commodity sets with different features:

- Baby&Toy: commodities in this dataset are mainly for the pregnant women, new mothers, toddlers and infants.
- Electronics: these commodities include digital products, computer peripherals, etc.
- Fashion: all kinds of clothes, such as blouse, shirt, shorts, trousers, skirt, dress, hat, etc.
- Food&Drink: the commodities about food and drink, e.g., green tea, wine, coffee, beef, cake, bread, fruit, etc. 


## Files in Each Dataset ##
The file structure in each dataset is the same. We will introduce these files one by one:

- scene-cate\_set.txt: the ground truth of scene mining. Each row is a scene and there are several categories in it.
- cate\_cate\_pair.txt: the relevance between categories. There are three elements in each row: category 1, category 2, and their relevance score.
- item\_brand\_pair.txt: the relationship between commodities and brands. There are three elements in each row: a commodity, a brand, and boolean which indicates whether whether the former's brand is the latter.
- item\_cate\_pair.txt: the relationship between commodities and categories. There are three elements in each row: a commodity, a category, and boolean which indicates whether the commodity belongs to the category.
- view\_item\_item\_pair.txt: the co-occur commodity pair in view sessions. There are three elements in each row: commodity 1, commodity 2, and normalized scores based on frequency.
- purchase\_item\_item_pair.txt: the co-occur commodity pair in orders. There are three elements in each row: commodity 1, commodity 2, and normalized scores based on frequency.
