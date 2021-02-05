#  Have I got a BEER for you?!
## A beer recommender for the adventurous beer drinker

My goal for this project was to create a beer recommendation system based "Wisdom of the Crowd", that is using K Nearest Neighbors based on item-item(user) similarity. In this case, the crowd that we're getting the wisdom from is in the form of the reviews that have been left of beers, taken from BeerAdvocate. 


## Table of Contents
    1. Data
    2. Exploratory Data Analysis (EDA)
    3. Building the model
    4. Selecting Model metrics
    5. Results
    6. Moving Forward

### 1. Data- 

To build the recommender, I needed some data about beer. There were three different data sets all pulled from Kaggle. 
The three data sets were: User Reviews, Beers, and Breweries. 
I needed to pull certain data from each data set by looking at the different columns.

User Reviews contained the following columns:

* Beer_id - id given to it from BeerAdvocate, the beer they're reviewing
* Username - what the reviewer goes by
* Date - the date the user left their review
* Text - any description they decided to leave
* Look - how good the user thinks it looks, scored from 1-5, 5 is best
* Smell - how the beer smells, scored same as look
* Taste - how the beer tastes ,scored same as smell
* Feel - how the beer feels in the user's mouth (also called mouthfeel), scored same as taste
* Overall - what the user 'thinks' the beers scores overall (not an average of the previous columns), scored same as feel
* Score - what the user thinks is the beers actual score, scored same as before


    ___While one would think that there isn't much diffence between 'Overall' and 'Score', it turns out there is. Every beer had a 'Score', but not every beer had an 'Overall'.___

Beers contained the following columns:

* ID - number assigned to that beer
* Name - name of the beer
* Brewery ID - number assigned to the brewery
* State - Where beer is made
* Country - Where beer is made
* Style - what type of beer it is (IPA, Stout, etc)
* Availability - is beer brewed year-round or only seasonally
* ABV - how much alcohol the beer has by volume %
* Notes - any special notes about the beer
* Retired - is the beer still produced or not

    ___Retired is something that I will have to add into my dataset in the future. Not as beers to recommend, but to still use for their 'score' to help with recommendations.___

Breweries contained the following columns:

* ID - number assigned to the brewery
* Name - name of the brewery
* City - where brewery is
* State - same as above
* Country - you get the picture
* Notes - extra info
* Types - What type of brewery it is (bar/eatery, brewpup, full on brewery)


### 2. EDA- 

After looking at the various columns of the three different data sets, I had to determine what was going to be used to build my model. While I would have liked to use more features (columns) like 'look', 'smell', and 'taste, due to the fact that so many of the reviews left by users had not filled them out, it meant that I wasn't able to include them as a scoring feature.

My final data set had pieces of all three, including:
* Username
* Score
* Beer name
* Brewery name
* Style


### 3. Building the Model

After creating my own data set, there were a couple of things that I had to use to be able to make my predictions (recommendations). Each beer and user had to have a BeerID and UserID created and linked, so that when looking at a beer, we can see who reviewed it. And when looking at a UserID, we're able to see what beers they reviewed. 



        username	    score	  beer_name	      brewery_name	                            style	        beerID	userID
    	Soloveitchik	4.67	Heady Topper	The Alchemist Brewery and Visitors Center	New England IPA	 5745	55539
    	Leffo	        5.00	Heady Topper	The Alchemist Brewery and Visitors Center	New England IPA	 5745	36902
    	liverust	    5.00	Heady Topper	The Alchemist Brewery and Visitors Center	New England IPA	 5745	93516

__The above is an example of looking at who reviewed a beer called Heady Topper, and what they rated it__


### 4. Selecting Model metrics

As stated at the beginning, beacause the goal for this project was to use the "wisdom of the crowd" to help with making a recommendation, using KNN to determine what beers would be best for the user, seemed to me the obvious way to go. However, in using the Surprise library for this, I found that there were quite a few different ways to make this happen. 

I decided to go with the KNN Baseline model that itself had different measuring metrics. I looked at 3 different ones: the Pearson Baseline, the Mean Squared Distance (MSD), and the Cosine Similarity. After looking at all three, I determined that it was the Pearson Baseline that gave me the best score, meaning that those beers recommended this way, would be those closest to what was entered by the user to find similarities to. 


### 5. Results

The results that I got were both promising and head-scratching at the same time. They were promising in that the beers that were recommended were good beers (I made sure of that when I set the threshold for the score at 4.0). They were head scratching in that I had expected to see more beers of the same style recommended. The reason for me expecting similar style is most likely due to personal bias, coming from a Bar Manager/ Bartender background. 


### 6. Moving Forward

After looking at the results, it became clear to me that for this recommender to become a truely viable product, that I would need to make some changes to it. One of the changes would have to be in the data itself: I need to remove beers that are no longer made from being able to be recommened but not removed from being a basis on making recommendations. I also need to include the location of the brewery, which will allow the user to make some assumptions about that brewery. Some assumptions are good: A brewery from New England will make a better New England Style IPA than a brewery in Southern California. And some assumptions are bad: All breweries in New England make beter beers than those in Southern California. Of course, in the end, I would like to try to develop this as an application, but that is a ways down the road.  

