import pandas as pd
import numpy as np
import copy
import pickle

from surprise import  KNNBasic, KNNWithMeans, NMF
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise import KNNBaseline, SVD
from surprise import get_dataset_dir
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD

from scipy import spatial

import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
import warnings
warnings.filterwarnings("ignore")

#loading in data
beers = pd.read_csv('~/den-19/capstone3/data/beers.csv')
#print(beers.columns)
beers_1= beers[['id', 'name', 'brewery_id', 'style']]
beers_1 = beers_1.rename(columns= {'id':'beer_id', 'name':'beer_name'})
beers_1= beers_1.sort_values(['beer_id'], ascending=True)
#print(beers_1.head(10))

reviews = pd.read_csv('~/den-19/capstone3/data/reviews.csv')
#print(reviews.columns)
reviews_1 = reviews[['beer_id', 'username', 'look','smell', 
            'taste','feel', 'overall', 'score']].sort_values(['beer_id'], ascending=True)
reviews_1 = reviews_1.reset_index()
reviews_1 = reviews_1.fillna(0)
reviews_1 = reviews_1.drop(['index'], axis = 1)
#print(reviews_1.head(10))


brewery = pd.read_csv('~/den-19/capstone3/data/breweries.csv')
#print(brewery.columns)
brewery_1 = brewery[['id', 'name']]
brewery_1 = brewery_1.rename(columns={'id':'brewery_id', 'name':'brewery_name'})
brewery_1 = brewery_1.sort_values(['brewery_id'])
#print(brewery_1.head(10))


#combine the databases to get what we need, then re-id both the beers and users

both= pd.merge(brewery_1, beers_1[['brewery_id', 'style','beer_name', 'beer_id']], on= 'brewery_id')

all_three = pd.merge(reviews_1, both[['beer_name', 'brewery_name', 'style', 'beer_id']], on = 'beer_id')
#still a little cleaning up to do
all_three_1 = all_three[['username', 'score', 'beer_name', 'brewery_name', 'style']]

#print(all_three_1.describe())
#this shows that 70% of beers have a ratting at or below 4.0
#we're going to round down to 4.0 as our threshold of recommendation

#we're going to plot a couple of things
data = all_three_1.beer_name.value_counts()
sns.set_context('talk')
sns.set_style('darkgrid')
plt.figure(figsize=(10,10))
ax= sns.histplot(data, bins= 50, kde=False, color='red', alpha=.5, log_scale= (True,True))
#second_ax = ax.twinx()
#sns.distplot(data, ax=second_ax, kde=True, hist=False)
#second_ax.set_yticks([])

ax.set_xlabel('Review Counts')
ax.set_ylabel('Num. of Beers')
ax.set_title('Histogram of Reivew Counts')

#now to find out where the beers fall and get rid of the low hanging fruit
grouped_beers = all_three_1.groupby('beer_name')
grouped_beers.count().sort_values(by='username', ascending= False).quantile(np.arange(0,1,.05))

grouped_beers.mean().sort_values(by='score', ascending=False).quantile(np.arange(0,1,.05))

avg_score = grouped_beers.mean()
#Here's where we get rid of those that were below 4.0 approx. the bottom 70%
below_avg = avg_score['score'] < 4.0
below_avg_count = len(avg_score[below_avg])

print('{} beers have an average score below 4.0'.format(below_avg_count))
print('A sub 4.0 avg score puts the beer within the bottom 70th percentile')

#bottom_20percent_rating = grouped_beers.mean().quantile(np.arange(0,.21,.02))
#bottom_20percent_counts = grouped_beers.count().quantile(np.arange(0,.21,.02))

#another plot, this one of user counts
data2 = all_three_1.username.value_counts()
sns.set_style('darkgrid')
sns.set_context(('talk'))
plt.figure(figsize=(10,10), dpi=250)
ax2 = sns.histplot(data2, bins=50, kde=False, color='red', alpha=.5, log_scale=(True, True))
#second_ax2 = ax2.twinx()
# sns.distplot(data2, ax=second_ax2, kde=True, hist=False)
# second_ax2.set_yticks([])

ax2.set_xlabel('Review Counts')
ax2.set_ylabel('Num. of Users')
ax2.set_title('Histogram of User Counts')

grouped_users= all_three_1.groupby('username')

grouped_users_count = grouped_users.count()

counts = [1,2,3,4,5,10,15,20]
for ct in counts:
    num_users = grouped_users_count[grouped_users_count['score'] <= ct].count()[0]
#     print('{} users rated {} or fewer beers'.format(num_users, ct))
# print('\n')

# print('Total Unique Users in this dataset: {}'.format(len(all_three_1.username.unique())))

subpar_beers_list= list(avg_score[below_avg].index)

ratings_count = grouped_beers.count()

#Here i'm setting beers that have ratings counts less than 20 to be called low amount of ratings

low_ratings_count = ratings_count[ratings_count['score'] < 20]
low_ratings_list = list(low_ratings_count.index)

unique_subpar_beers = set(subpar_beers_list)
unique_low_ratings_beers = set(low_ratings_list)
overlap = unique_subpar_beers.intersection(unique_low_ratings_beers)
# print('Number of beers in bottom 40% of avg rating: {}'.format(len(unique_subpar_beers)))
# print('Number of beers in bottom 40% of review counts: {}'.format(len(unique_low_ratings_beers)))
# print('Number of beers in both categories: {}'.format(len(overlap)))


#Here the beers that are in the bottom of both categories are removed
df1 = all_three_1[~all_three_1.beer_name.isin(subpar_beers_list)]
df2 = df1[~df1.beer_name.isin(low_ratings_list)]
all_three_2 = copy.deepcopy(df2)

# print(all_three_2.info())

# print('Original number of unique beers: {}'.format(len(all_three_1.beer_name.unique())))
# print('Revised number of unique beers: {}'.format(len(all_three_2.beer_name.unique())))


#creating a new beerID for each beer
grouped_name = all_three_2.groupby('beer_name')
temp_df = grouped_name.count()
temp_df_idx = pd.DataFrame(temp_df.index)

temp_df_idx['beerID'] = temp_df_idx.index
dict_df = temp_df_idx[['beerID', 'beer_name']]

desc_dict = dict_df.set_index('beer_name').to_dict()
new_dict = desc_dict['beerID']

all_three_2['beerID']= all_three_2.beer_name.map(new_dict)

#creating a new userID for each user
grouped_user = all_three_2.groupby('username')

temp_df_user = grouped_user.count()
temp_df_user_idx = pd.DataFrame(temp_df_user.index)

temp_df_user_idx['userID'] = temp_df_user_idx.index
dict_df_user = temp_df_user_idx[['userID', 'username']]

desc_dict_user = dict_df_user.set_index('username').to_dict()
new_dict_user = desc_dict_user['userID']

all_three_2['userID'] = all_three_2.username.map(new_dict_user)

# print(all_three_2.iloc[0].beer_name)
# print(all_three_2.iloc[0].username)

def read_item_name():
    # two maps to convert raw ids into beer names and vice versa

    file_name = dict_df
    raw_id_to_name = {}
    name_to_raw_id = {}

    #there were 13274 unique beers after removing those w/ low rating and review counts
    unique_beers = len(all_three_2.beer_name.unique())
    for i in range (unique_beers):
        line = file_name.iloc[i]
        raw_id_to_name[line[0]] = line[1]
        name_to_raw_id[line[1]]= line[0]

    return raw_id_to_name, name_to_raw_id
    


def get_recommendation(beer_name, k_):
    
    '''
    input a beer and get k-num recommendations
       that are based on similarity

       input = string, int
       output = string
    ''' 

    output = []
    beer = str(beer_name)
    
    #read the maps raw id -->beer name, and vice versa
    raw_id_to_name, name_to_raw_id = read_item_name()

    #get the inner id of the beer
    beer_input_raw_id = name_to_raw_id[beer]
    beer_input_inner_id = algo.trainset.to_inner_iid(beer_input_raw_id)

    K = k_

    #get the inner ids of the nearest neighbors of the beer
    beer_input_near_neigh = algo.get_neighbors(beer_input_inner_id, k=K)

    beer_input_near_neigh = (algo.trainset.to_raw_iid(inner_id) for inner_id in beer_input_near_neigh)

    beer_input_near_neigh = (raw_id_to_name[rid] for rid in beer_input_near_neigh)

    for beer_ in beer_input_near_neigh:
        output.append(beer_)

    return output



#train the algorithm to compute the similarities bewteen items
reader= Reader(rating_scale=(1,5))
data= Dataset.load_from_df(all_three_2[['userID', 'beerID', 'score']],reader)
trainset, testset = train_test_split(data, test_size = .20)

trainset = data.build_full_trainset()
sim_options = {'name':'pearson_baseline', 'user_based': False}
algo= KNNBaseline(sim_options = sim_options)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)


with open('algo.pkl', 'wb') as f:
    pickle.dump(algo, f)


with open('algo.pkl', 'rb') as m:
    algo_model = pickle.load(m)

algo_model.test(testset)

############

#This evaluates the performance using Surprise's evaluation module, but couldn't get it to import
#evaluate(algo, data, measures = ['RMSE', 'MAE'])

#top 20 most rated beers
grouped_beer_names = all_three_2.groupby('beer_name')
grouped_beer_names.count().sort_values(by='username', ascending= False)[0:20].index.tolist()

# #top 20 highest rated beers
grouped_beer_names.mean().sort_values(by='score', ascending = False)[0:20].index.tolist()


print('The 20 nearest neighbors of Heady Topper are:')
print(get_recommendation('Heady Topper',20 ))


# print('The 20 nearest neighbor to 90 Minute IPA are:')
# print(get_recommendation('90 Minute IPA', 20))

top_20_rated = all_three_2.groupby('beer_name').count().sort_values(by= 'username', ascending = False)[0:20].index.tolist()
top_20_rated = set(top_20_rated)

top_20_scores = all_three_2.groupby('beer_name').mean().sort_values(by='score', ascending = False)[0:20].index.tolist()
top_20_scores = set(top_20_scores)

dfh90 = set(get_recommendation('90 Minute IPA', 20))
gmviet = set(get_recommendation('Good Morning', 20))
kbbs = set(get_recommendation('Kentucky Brunch Brand Stout', 20))
tha = set(get_recommendation('Two Hearted Ale', 20))

ebi = set(get_recommendation('Enjoy By IPA', 20))
# print('The 20 nearest neighbors of Enjoy By IPA:')
# print(ebi)



all_three_2_pivot = all_three_2.pivot_table(index = 'username', columns = 'beer_name', values= 'score').fillna(0)

Transposed = all_three_2_pivot.values.T
#print(Transposed.shape)

def explained_vary(list_n_components):
    
    '''
    input = list of integers
    output = list of tuples showing n_components and it's explained 
    variance ratio
    '''

    out= []
    
    for num in list_n_components:
        SVD = TruncatedSVD(n_components = num, random_state= num)
        SVD.fit_transform(Transposed)
        evar = np.sum(SVD.explained_variance_ratio_)
        t = (num, evar)
        out.append(t)
        
    return out

n_comp = [10,20,50,100,200,300,400]
explained_variance = explained_vary(n_comp)

[print(i) for i in explained_variance]
x,y = zip(*explained_variance)
plt.scatter(x,y)

SVD_400 = TruncatedSVD(n_components=200, random_state= 42)
matrix_400 = SVD_400.fit_transform(Transposed)
print(matrix_400.shape)

#correlation/similarity matrix
corr_400 = np.corrcoef(matrix_400)
print(corr_400.shape)

#name of all beers
beer_rec_names_400 = all_three_2_pivot.columns
#list of all beer names
beer_rec_list_400 = list(beer_rec_names_400)

def svd_400_recomm(string, n):
    
    '''
    function that returns top n-num of recommendations
    based on input of beer name and n

    inputs: string(name of beer) --> string
            n (num recommendations) --> int
    '''

    #we have to get the index of the beer name form list of
    #all beers in the training data
    get_index = beer_rec_list_400.index(string)

    similarities = corr_400[get_index]

    closest = []
    for idx, coeff in enumerate(similarities):
        closest.append((beer_rec_list_400[idx], coeff))
    
    closest.sort(key=lambda x: x[1], reverse = True)

    out= []
    for i in range (1, n+1):
        out.append(closest[i][0])
    return out

#looking at the top 20 from a beer earlier
print(svd_400_recomm('Enjoy By IPA', 50))

#latent factors method
A = set(svd_400_recomm('Enjoy By IPA', 50))
#neighborhood method
B = set(get_recommendation('Enjoy By IPA', 50))

print('common beers: {}'.format(A.intersection(B)))
print('number of common beers: {}'.format(len(A.intersection(B))))

def compare_recomm(name_list, n):
    results = []
    for idx, name in enumerate(name_list):
        svd = set(svd_400_recomm(name, n))
        knn = set(get_recommendation(name, n))
        common = len(svd.intersection(knn))
        tup = (idx, common)
        results.append(tup)

    x,y = zip(*results)
    plt.scatter(x,y)
    plt.xlabel('Beer No.')
    plt.ylabel('Common Recommendations')
    plt.show()


#select every 40 beers from the highest avg rating to lowest avg rating
grouped = all_three_2.groupby('beer_name')
namelist= grouped.mean().sort_values(by= 'score', ascending=False)[::40].index.tolist()

print(compare_recomm(namelist, 100))




