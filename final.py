"""
Define imports

"""

import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, KNNBasic, NormalPredictor,BaselineOnly,KNNWithMeans,KNNBaseline
from surprise import SVD, SVDpp, NMF, SlopeOne, CoClustering
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise import accuracy

from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
#nltk.download('averaged_perceptron_tagger')
from sklearn.feature_extraction import text
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet as wn
import string

import random
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None  # default='warn'

"""
Function definitions

These are the functions that will be used throughout the project

The sanitize, create_soup, fetchSimilarityMatrix, and content_recommender 
functions are borrowed and modified from functions used in DS775 - Prescriptive Analytics

"""

def sanitize(x):
    """
    This function sanitizes the imput text to make it more useful for the soup
    """
    if isinstance(x, list): 
        return [str.lower(i.replace(" ", "")) for i in x] #If the input data is a list, return a string with removed spaces
    else:
        if isinstance(x, str):
            return str.lower(x) #Apply lower case to all text
        else:
            return ''
   

def create_soup(x):
    """
    This function creates the soup used in the content-based portion of the recommender.
    """
    soup =  x[' Name (Product)']\
            + ' ' + x[' Description 2 (Product)'] \
            + ' ' + x[' Buy Line (Product)'] \
            + ' ' + x[' Price Line (Product)'] \
            + ' ' + x[' GL Type (Product)'] \
            + ' ' + x[' Code (Category Level 1)'] \
            + ' ' + x[' Name (Price Line)'] \
            + ' ' + x[' Name (GL Type)'] \
            + ' ' + x[' Code (Category Level 2)']
    
    return soup      

def fetchSimilarityMatrix(df, soupCol, vectorizer, vectorType='Tfidf'):
    '''
    This function gets the similarity matrix using the soup data.
    
    df: input dataframe
    soupCol: title of the column containing the soup
    vectorizer: an initialized vectorizer
    vectorType: either 'Tfidf' or 'Count' 

    Returns a similarity matrix used for content-based recommendation
    '''

    
    df[soupCol] = df[soupCol].fillna('') #Removes NA from the soup column if there are any instances
    nmatrix = vectorizer.fit_transform(df[soupCol]) #creates a matrix from the input data frame

    #apply the appropriate vectorizer
    if(vectorType=='Tfidf'):
        print('Using Linear Kernel (Tfidf)')
        sim =linear_kernel(nmatrix, nmatrix)
    else:
        print('Using Cosine_similarity')
        sim = cosine_similarity(nmatrix, nmatrix)
    return(sim)

def content_recommender(df, seed, seedCol, sim_matrix,  topN=2):
    """
    This function is the content-based portion of the recommender system.

    Functional steps:
    1. Get the indices based off the seedCol
    2. Obtain the index of the item that matches the seed (product)
    3. Get the pairwsie similarity scores of all items and convert to tuples
    4. Delete the seed item that was passed in so it doesn't get recommended to itself (similarity = 1)
    5. Sort the items based on the similarity scores
    6. Get the scores of the top-n most similar items.
    7. Get the item indices so they can be extracted
    8. Return the topN most similar items

    """

    df.reset_index(inplace=True)

    indices = pd.Series(df.index, index=df[seedCol]).drop_duplicates()
    
    idx = indices[seed]
    #print(df.loc[df[' Code (Product)'] == seed]) <- used for testing
    
    sim_scores = list(enumerate(sim_matrix[idx]))
    
    del sim_scores[idx]
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[:topN]
    
    item_indices = [i[0] for i in sim_scores]
    
    return df.iloc[item_indices]

def hybrid(user, product, predCol, algorithm, N):
    '''
    Parameters
    user: the user for whom we are making predictions (customer, identified by customer code)
    predCol: the column in the dataframe that will be used to make predictions (product code)
    algorithm: a trained Surprise model that will be used for making predictions
    N: the number of predictions to return

    Returns
    a pandas dataframe containing everything in the contentRecs dataframe, plus the estimated rating for the user requested
    '''

    current_gl, filter_gls = choose_vertical(sales_no_dup, product) # Split input data frame into inter- and intra-GL for cross-selling
    #print(current_gl.head()) <- used for testing

    #Initialize a vector for each case
    count1 = CountVectorizer(lowercase=True, stop_words='english') 
    count2 = CountVectorizer(lowercase=True, stop_words='english')

    # Get the similarity matrix for the current GL and the related GLs
    sim1 = fetchSimilarityMatrix(current_gl, 'soup', count1, 'Count')
    sim2 = fetchSimilarityMatrix(filter_gls, 'soup', count2, 'Count')
    
    
    """
    Find the top 50 products within the GL based on content, calculate the estimated rating of those products using the input 
    algorithm, sort those results by their rating, and then choose the top 3. These are the top 3 items recommended within the 
    seed product's GL Code (product group)
    """
    gl_results = content_recommender(current_gl, product, predCol, sim1, 50)
    #print(gl_results.head(10)) <- used for testing
    gl_results['est_rating'] = gl_results.apply(lambda x: algorithm.predict(user, x[predCol]).est, axis=1)
    gl_results = gl_results.sort_values('est_rating', ascending=False)
    gl_results_top = gl_results.head(3)
    
    """
    Find the top 50 products within the  *related* GLs based on content, calculate the estimated rating of those products using the input 
    algorithm, sort those results by their rating, and then choose the top 3. These are the top 3 items recommended in *related* product groups.
    """

    results = content_recommender(filter_gls, product, predCol, sim2, 50)
    #print(results.head(10)) <- used for testing
    results['est_rating'] = results.apply(lambda x: algorithm.predict(user, x[predCol]).est, axis=1)
    results = results.sort_values('est_rating', ascending=False)
    results_top = results.head(3)
    
    frames = [gl_results_top, results_top] # create a data frame from the top 3 results for each portion of the recommender

    final_results = pd.concat(frames)
    
    return final_results.head(N) #return the finalN number of results

def choose_vertical(df, product):
    """
    This function chooses the "vertical" or area of the business in which the product recommendations will take place

    Since the business is functionally divided between industrial and construction sides, there is very little crossover between
    sales in these two areas. This function helps the recommender system's accuracy in eliminated products that may have similar soup content
    but are otherwise poor recommendations.
    """

    current_GL =  df.loc[df[' Code (Product)'] == product, ' Name (GL Type)'].item() # determine the current GL (product group)
    gl_specific = df[df[' Name (GL Type)'] == str(current_GL)]

    #if the current GL is in the construction side of the business, eliminate all the industrial GL codes from the data frame
    if current_GL in construction_GLs_lower:
        for i in industrial_GLs_lower:
            df = df[df[' Name (GL Type)'] != i]
    #if the current GL is in the industrial side of the business, eliminate the construction GL codes from the data frame.
    elif current_GL in industrial_GLs_lower:
        #additionally, for industrial GLs, remove the GL itself if the product is not in that GL
        df = df.loc[~((df[' Name (GL Type)'] == current_GL) & (df[' Code (Product)'] != product)),:]
        for i in construction_GLs_lower:
            df = df[df[' Name (GL Type)'] != i]

    #return the data frame with intra-GL products and the one for related GLs        
    return gl_specific, df

def pick_rec_algo(dict_of_rmse):
    v = list(dict_of_rmse.values())
    k = list(dict_of_rmse.keys())
    best_rmse = k[v.index(min(v))]
    return best_rmse


"""
Import dataset and clean

"""

sales = pd.read_csv('October_sales.csv', index_col=False) #import the dataset from CSV

sales = sales.astype({' Code (Product)': 'string'})
sales = sales.astype({' Value': float})

#Eliminate junk values from the transactional data
sales[' Code (Category Level 1)'] = sales[' Code (Category Level 1)'].map({'###': ''})
sales[' Code (Category Level 2)'] = sales[' Code (Category Level 2)'].map({'###': ''})

#GLs within the construction side of the business
construction_GLs = ["Electrical Supplies"
                  , "Wire & Cable"
                  , "Enclosures"
                  , "Lighting Sales"]

#GLs that do not have value for recommendations (services, accounting entries, custom solutions)
not_needed_GLs = ["AB Contract Services"
                , "AB MV & MCC Products"
                , "AB Transactional Ser"
                , "Project Solutions"
                , "Commission Rep"
                , "Technical Labor Serv"
                , "Distribution Equip."
                , "Engraving"
                , "ASP"
                , "Clean Energy"
                , "Digital Solutions"
                , "Jobsite Services"]

#GLs within the industrial side of the business
industrial_GLs = ["AB Indust Controls"
                , "AB Sensing Products"
                , "AB Motion Controls"
                , "Data Communications"
                , "AB Programmable Cont"
                , "AB Drive Products"
                , "AB Safety Products"
                , "AB EOI Comp & Soft"
                , "Pneumatics/Hydraulic"
                , "Network Solutions"
                , "Process Solutions"]

#create lists of the industrial and construction GLs as lowercase
industrial_GLs_lower = [x.lower() for x in industrial_GLs]
construction_GLs_lower = [x.lower() for x in construction_GLs]

#remove strings that may poison the soup that have no value for similarity
strings_to_remove = ["LABOR", "MISC", "STOREROOM", "PREFAB"]
sales = sales[~sales[' Name (Product)'].str.contains("|".join(strings_to_remove), na=False)]

#remove entries where the value is less than zero (returns, accounting adjustments)
sales = sales[sales[' Value'] > 0]

#print(sales.head()) <- used for testing

#create a dataframe of products for use in the content-based filter by removing duplicates
sales_no_dup = sales.drop_duplicates(subset=[' Code (Product)'])

#remove the non-needed GLs
for i in not_needed_GLs:
    sales = sales[sales[' Name (GL Type)'] != i]

#remove unnecessary columns from the data frame for collaborative filtering and group by count (higher count means higher rating)
collab_sales = sales[[' Code (Ship To)', ' Code (Product)', ' Value']]
grouped_sales = collab_sales.groupby([' Code (Ship To)', ' Code (Product)'], as_index=False).count()

#Scale the Value column to reduce skew
scaler = MinMaxScaler(feature_range=(1,10))
grouped_sales[' Value'] = scaler.fit_transform(grouped_sales[' Value'].values.reshape(-1,1))

"""
Make soup and create similarity matrix

"""

soup_columns = [' Name (Product)'
                , ' Description 2 (Product)'
                , ' Buy Line (Product)'
                , ' Price Line (Product)'
                , ' GL Type (Product)'
                , ' Name (Price Line)'
                , ' Name (GL Type)'
                , ' Code (Category Level 1)'
                , ' Code (Category Level 2)'
                , ' Code (Catalog #)']

#create the soup
for i in soup_columns:
    sales_no_dup[i] = sales_no_dup[i].apply(sanitize)

#remove strings that identify vendor that poison predctions
sales_no_dup['soup'] = sales_no_dup.apply(create_soup, axis=1)
sales_no_dup['soup'] = sales_no_dup['soup'].str.replace('a-b', '')
sales_no_dup['soup'] = sales_no_dup['soup'].str.replace('dmi', '')


"""
Begin collaborative filter

"""

rmse_dict = {}

#define a seed for testing
our_seed = 10 

reader = Reader(rating_scale=(1,10))

data = Dataset.load_from_df(grouped_sales, reader)

raw_ratings = data.raw_ratings

random.seed(our_seed)
np.random.seed(our_seed)
random.shuffle(raw_ratings)

#Split dataset into training and test datasets with a 9:1 ratio
threshold = int(.9 * len(raw_ratings))
train_raw_ratings = raw_ratings[:threshold]
test_raw_ratings = raw_ratings[threshold:]

data.raw_ratings = train_raw_ratings

#Define the hyperparameter grid that will be used for grid search on the KNN algorithm
param_grid = {'k': [3,15], 'min_k': [1,3]} 

#Perform a grid search on the parameter grid with 5-fold cross validation
grid_search = GridSearchCV(KNNBasic, param_grid, measures=['rmse'], cv=5)
grid_search.fit(data)

#Extract best parameters
knn_model = grid_search.best_estimator['rmse']

#Print best parameters
print(grid_search.best_params['rmse'])

#Build a full testset and train the model on the dataset with the best hyperparameters
trainset = data.build_full_trainset()
knn_model.fit(trainset)

# Compute biased accuracy on trainset 
predictions = knn_model.test(trainset.build_testset())
print(f'Biased accuracy on train = {accuracy.rmse(predictions)}')


# Compute unbiased accuracy on testset
testset = data.construct_testset(test_raw_ratings)
predictions = knn_model.test(testset)
print(f'Unbiased accuracy on test = {accuracy.rmse(predictions)}')

knn_cv = cross_validate(knn_model, data, measures=['RMSE'], cv=5, verbose=True)

knn_RMSE = np.mean(knn_cv['test_rmse'])

rmse_dict['knn'] = knn_RMSE

#Print RSME for evaluation
print(f'\nThe RMSE across five folds for KNN was {knn_RMSE}')

#Define a Single Value Decomposition object
svd = SVD()

random.seed(our_seed)
np.random.seed(our_seed)

#Define the hyperparameter grid that will be used for grid search on the SVD algorithm
param_grid_svd = {'n_epochs': [20, 40], 'lr_all': [0.004, 0.006]} #Number of epochs from 20 to 40 will be searched, as well as learning rates of 0.004 to 0.006

#Perform agrid search on the SVD algorithm to find best hyperparameters
grid_search_svd = GridSearchCV(SVD, param_grid_svd, measures=['rmse'], cv=5)
grid_search_svd.fit(data)

#Print the best hyperparameters for analysis
print(grid_search_svd.best_params['rmse'])

#Extract the best hyperparameter values from the grid search
svd = grid_search_svd.best_estimator['rmse']

#Perform 5 fold cross validation on the SVD object
svd_cv = cross_validate(svd, data, measures=['RMSE'], cv=5, verbose=True)

svd_RMSE = np.mean(svd_cv['test_rmse'])

rmse_dict['svd'] = svd_RMSE

#Print RSME for evaluation
print(f'\nThe RMSE across five folds for SVD was {svd_RMSE}')

#Build a full model on the data
trainset = data.build_full_trainset()
svd.fit(trainset)

#Define CoClustering object
co_cluster = CoClustering()

random.seed(our_seed)
np.random.seed(our_seed)

#Define hyperparameter grid. Number of user clusters from 3 to 10 and number of item clusters from 3 to 10
param_grid_cc = {'n_cltr_u': [3, 10], 'n_cltr_i': [3, 10]}

#Perform a grid search to find best hyperparameters
grid_search_cc = GridSearchCV(CoClustering, param_grid_cc, measures=['rmse'], cv=5)
grid_search_cc.fit(data)

#Print the best parameters
print(grid_search_cc.best_params['rmse'])

#Extract the best hyperparameters from the grid search object
cc = grid_search_cc.best_estimator['rmse']

#Perform 5-fold cross validation on the CoClustering model
cc_cv = cross_validate(cc, data, measures=['RMSE'], cv=5, verbose=True)

cc_RMSE = np.mean(cc_cv['test_rmse'])

rmse_dict['cc'] = cc_RMSE

#Print RSME for evaluation
print(f'\nThe RMSE across five folds for CoCLustering was {cc_RMSE}')

#Build a full model on the data
trainset_cc = data.build_full_trainset()
cc.fit(trainset_cc)

"""
This final section is for extracting the recommendations

When deployed to a production environment like a website, this function 
would effectively be in a loop to avoid re-training the input data for
every interaction with the website, but for the purposes of this project 
is used to demonstrate the functionality of the system.

"""

best_algo = exec(pick_rec_algo(rmse_dict))

final_recs = hybrid("17064", "54738", ' Code (Product)', svd, 10)
#final_recs.to_csv('recs.csv') <- used for testing
print(final_recs)
