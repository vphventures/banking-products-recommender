# Imports

import pandas as import pd
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR
import random
from sklearn import metrics
from sklearn.preprocessing import MaxAbsScaler

# Library settings

pd.set_option('display.width', 600)
np.set_printoptions(threshold=np.nan)

# Connection settings

engine = create_engine('oracle://username:password@db')
oracle_con = engine.connect()

# Data extraction

def data_extraction(oracle_con):
    sql = """
        select * from transactions
        """
    retail_data = pd.read_sql(sql, oracle_con)
    df_logging(retail_data, 1, 'retail_data')
    return retail_data

def df_logging(df, order, filename):
    pd.Dataframe(df).to_csv('path')

# Data cleaning

def data_cleaning(retail_data):
    cleaned_retail = retail_data.loc[pd.isnull(retail_data.siebel_id) == False]

    cleaned_retail['prod_code'] = cleaned_retail.prod_code.astype(str)

    df_logging(cleaned_retail, 2, 'cleaned_retail')
    # Only get unique item/description pairs
    item_lookup = cleaned_retail[['prod_code', 'prod_desc']].drop_duplicated() 
    # Encode as string for future lookup ease
    item_lookup['prod_code'] = item_lookup.prod_code.astype(str)

    df_logging(retail_data, 3, 'item_lookup')
    # Convert to int for customer ID
    cleaned_retail['siebel_id'] = cleaned_retail.siebel_id.astype(int)
    # Get rid of unneccesary info
    cleaned_retail = cleaned_retail[['prod_code', 'qty', 'siebel_id']]
    df_logging(retail_data, 4, 'cleaned_retail')
    # Group together
    grouped_cleaned = cleaned_retail.groupby(['siebel_id', 'prod_code']).sum().reset_index()
    # Replace a sum of zero purchases with a one to indicated purchased
    grouped_cleaned.qty.loc[grouped_cleaned.qty == 0] = 1
    # Only get customers where purchase totals were positive
    grouped_purchased = grouped_cleaned.loc[grouoed_cleaned['qty'] >= 0]
    df_logging(retail_data, 5, 'grouped_purchased')

    item_lookup_df = item_lookup.set_index('prod_code')

    item_lookup_df.to_sql('item_lookup', oracle_con, if_exists='replace', dtype={'prod_code': VARCHAR(250), 'prod_desc': VARCHAR(250)})

    return grouped_purchased, item_lookup

def create_matrix(grouped_purchased):
    # Get unique customers
    customers = list(np.sort(grouped_purchased.siebel_id.unique()))
    # Get unique products that were purchased
    products = list(grouped_purchased.prod_code.unique())
    # All purchases
    qty = list(grouped_purchased.qty)

    # Vytvori radky pro matici, kdy pro kazdy radek znovu oindexuje customery od 0
    rows = grouped_purchased.siebel_id.astype('category', categories=customers).cat.codes
    df_logging(retail_data, 6, 'rows')
    # Vytvori sloupce pro matici, kdy pro kazdy produkt v radcich znovu oindexuje customery od 0
    cols = grouped_purchased.prod_code.astype('category', categories=products).cat.codes
    df_logging(retail_data, 7, 'cols')

    # Vytvori matici ale neni to uplne dataframe, co radek to bunka v matici s hodnotou v bunce
    # (0, 0) 6
    # (0, 1) 12
    purchased_sparse = sparse.csr_matrix((qty, (rows, cols)), shape=(len(customers), len(products)))

    # Pocet kombinace v matici X krat Y = 13212312 moznosti vazeb v matici
    # Number of possible interactions in the matrix
    matrix_size = purchased_sparse.shape[0] * purchased_sparse.shape[1]

    # Pocet bunek, kde quantity neni 0
    # Number of items interacted with
    num_purchases = len(purchases_sparse.nonzero()[0])

    # Mira vyplnenosti matice, 95% je max pro collaborative filtering
    sparsity = 100 * (1-(num_purchases/matrix_size))

    print('sparsity', sparsity)

    cust_df = pd.Dataframe(customers)
    cust_df.index_name = 'cust_id'
    cust_df.columns = ['siebel_id']
    cust_df.to_sql('customers', oracle_con, if_exists="replace")

    return purchased_sparse, customers, products

def make_train(ratings, pct_test = 0.2):
    """
    This function will take in the original user-item matrix and mask a percentage of the original ratings
    where a user-item interaction has taken place for use as a test set.
    The test set will contain all of the original ratings, while the training set replace the specified
    percenttage of them with a zero in the original ratings matrix

    parameters:

    ratings - original ratings matrix from which you want to generate a train/test set. Test is just a complete 
    copy of original set. This is in the form of a sparce csr_matrix

    pct_set - the percentage of user-item interactions where an interaction took place, that you want to
    mask in the training set for later comparison to the test set, which contains all of the original ratings

    returns:

    training_set - the altered version of the original data with certain percentage of the user item pairs
    that originaly had interaction set back to zero.

    test_set - a copy of the original ratings matrix, unaltered, so it can be used to see how the rank order
    compare with the actual interactions

    user_inds - from the randomly selected user-item indices, which user rows were altered in the training data.
    This will be necessary later when evaluating the performance via AUC.

    """
    # Make copy of the original set to be the test set
    test_set = raings.copy()
    
    # Testovaci matici olabluje hodnoty nakupi jako 1
    test_set[test_set != 0] = 1
    
    # Make a copy of the original data we can alter as our training set
    training_set = ratings.copy()

    # Vytvori pole indexu matice ([x1, x2],[y1,y2]) bunek, ktere maji pozitivni nakup
    # Find indices in the ratings data where an interaction exists
    nonzero_inds = training_set.nonzero()

    # Spoji ([x1, x2],[y1,y2])  na (x1,y1),(x2,y2) proste je sparuje
    # Zip these pairs together of user, item index into list
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))
    # Set random seed to zero for reproducibility
    random.seed(0)
    # Test sample velikost = % * dimenze nonzero paru a pak zaokrouhli nahoru
    num_sample = int(np.ceil(pct_test * len(nonzero_pairs)))

    # Vezme puvodni populaci nonzero pairs a nahodne vybere pary a da je do sample mnoziny, puvodni populaci necha netknutou
    # Sample a random number of user item pairs without replacement
    samples = random.sample(nonzero_pairs, num_sample)

    # Ziska ze samplu indexy customeru, vezme se prvni hodnota z paru
    # Get user row indices
    user_inds = [index[0] for index in samples]

    # Ziska ze sampli indexy produktu, vezme se druha hodnota z paru
    # Get the item column indices
    item_inds = [index[1] for index in samples]

    # S pomoci indexu sampli se premeni bunky traning matice na 0
    # Assign all of the randomly chosen user item pairs to zero
    training_set[user_inds, item_inds] = 0

    # Ciste optimalizacni vec, zakryte sample bunky smaze z traninng setu - sparce matice
    # Get rid of zeros in sparse array storage after update to save space
    training_set.eliminate_zeros()

    # Export training matice bez nul, testovaci matice a seznam sample bunek
    product_users_altered = list(set(user_inds)) 

    return traning_set, test_set, product_users_altered

def implicit_weighted_als(traning_set, lambda_val, alpha, iterations, rank_size, seed):
    '''
    Implicit weighted ALS taken from Hu, Koren, and Volinsky 2008. Designed for alternating least squares and implicit
    feedback based collaborative filtering. 
    
    parameters:
    
    training_set - Our matrix of ratings with shape m x n, where m is the number of users and n is the number of items.
    Should be a sparse csr matrix to save space. 
    
    lambda_val - Used for regularization during alternating least squares. Increasing this value may increase bias
    but decrease variance. Default is 0.1. 
    
    alpha - The parameter associated with the confidence matrix discussed in the paper, where Cui = 1 + alpha*Rui. 
    The paper found a default of 40 most effective. Decreasing this will decrease the variability in confidence between
    various ratings.
    
    iterations - The number of times to alternate between both user feature vector and item feature vector in
    alternating least squares. More iterations will allow better convergence at the cost of increased computation. 
    The authors found 10 iterations was sufficient, but more may be required to converge. 
    
    rank_size - The number of latent features in the user/item feature vectors. The paper recommends varying this 
    between 20-200. Increasing the number of features may overfit but could reduce bias. 
    
    seed - Set the seed for reproducible results
    
    returns:
    
    The feature vectors for users and items. The dot product of these feature vectors should give you the expected 
    "rating" at each point in your original matrix. 
    '''
    
    # first set up our confidence matrix
    
    conf = (alpha*training_set) # To allow the matrix to stay sparse, I will add one later when each row is taken 
    # and converted to dense. 
    num_user = conf.shape[0]
    num_item = conf.shape[1] # Get the size of our original ratings matrix, m x n
    
    # initialize our X/Y feature vectors randomly with a set seed
    rstate = np.random.RandomState(seed)
    
    # Vytvori mezimatici s latentnimi featues s pocatecnimi hodnotami pro user dimenzi
    X = sparse.csr_matrix(rstate.normal(size = (num_user, rank_size))) # Random numbers in a m x rank shape
    df_logging(X.toarray(), 12, 'X')
    # Vytvori mezimatici s latentnimi featues s pocatecnimi hodnotami pro item dimenzi
    Y = sparse.csr_matrix(rstate.normal(size = (num_item, rank_size))) # Normally this would be rank x n but we can 
                                                                 # transpose at the end. Makes calculation more simple.
    df_logging(X.toarray(), 13, 'Y')
    # Vytvori sparse matici s 1 na diagonale o velikosti users x users
    X_eye = sparse.eye(num_user)
    # Vytvori sparse matici s 1 na diagonale o velikosti item x item
    Y_eye = sparse.eye(num_item)
    # Matice s hodnotama proti prefitovani?
    lambda_eye = lambda_val * sparse.eye(rank_size) # Our regularization term lambda*I. 
    
    # We can compute this before iteration starts. 
    
    # Begin iterations
   
    for iter_step in range(iterations): # Iterate back and forth between solving X given fixed Y and vice versa
        # Compute yTy and xTx at beginning of each iteration to save computing time
        # Transponovanou matici MxF s nahodnymi cisly vynasobi puvodni matici MxD
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)
        # Being iteration to solve for X based on fixed Y
        for u in range(num_user):
            # Pro kazdy radek - usera v konfidencni matici sebere radek a udelaz z nej vector
            conf_samp = conf[u,:].toarray() # Grab user row from confidence matrix and convert to dense
            pref = conf_samp.copy() 
            # Vector se binerarizujea prejmenuje na preferencni user vector
            pref[pref != 0] = 1 # Create binarized preference vector 
            # Vytvori items matici kde na diagonale je confidence kazdeho itemu?
            CuI = sparse.diags(conf_samp, [0]) # Get Cu - I term, don't need to subtract 1 since we never added it 
            yTCuIY = Y.T.dot(CuI).dot(Y) # This is the yT(Cu-I)Y term 
            yTCupu = Y.T.dot(CuI + Y_eye).dot(pref.T) # This is the yTCuPu term, where we add the eye back in
                                                      # Cu - I + I = Cu
            X[u] = spsolve(yTy + yTCuIY + lambda_eye, yTCupu) 
            # Solve for Xu = ((yTy + yT(Cu-I)Y + lambda*I)^-1)yTCuPu, equation 4 from the paper  
        # Begin iteration to solve for Y based on fixed X 
        for i in range(num_item):
            conf_samp = conf[:,i].T.toarray() # transpose to get it in row format and convert to dense
            pref = conf_samp.copy()
            pref[pref != 0] = 1 # Create binarized preference vector
            CiI = sparse.diags(conf_samp, [0]) # Get Ci - I term, don't need to subtract 1 since we never added it
            xTCiIX = X.T.dot(CiI).dot(X) # This is the xT(Cu-I)X term
            xTCiPi = X.T.dot(CiI + X_eye).dot(pref.T) # This is the xTCiPi term
            Y[i] = spsolve(xTx + xTCiIX + lambda_eye, xTCiPi)
            # Solve for Yi = ((xTx + xT(Cu-I)X) + lambda*I)^-1)xTCiPi, equation 5 from the paper
    # End iterations

    user_vecs = pd.DataFrame(X.toarray())
    item_vecs = pd.DataFrame(Y.toarray())

    return X, Y.T # Transpose at the end to make up for not being transposed at the beginning. 
    # Y needs to be rank x n. Keep these as separate matrices for scale reasons. 

def auc_score(predictions, test):
    '''
    This simple function will output the area under the curve using sklearn's metrics. 
    
    parameters:
    
    - predictions: your prediction output
    
    - test: the actual target result you are comparing to
    
    returns:
    
    - AUC (area under the Receiver Operating Characterisic curve)
    '''
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)

def calc_mean_auc(training_set, altered_users, predictions, test_set):
    '''
    This function will calculate the mean AUC by user for any user that had their user-item matrix altered. 
    
    parameters:
    
    training_set - The training set resulting from make_train, where a certain percentage of the original
    user/item interactions are reset to zero to hide them from the model 
    
    predictions - The matrix of your predicted ratings for each user/item pair as output from the implicit MF.
    These should be stored in a list, with user vectors as item zero and item vectors as item one. 
    
    altered_users - The indices of the users where at least one user/item pair was altered from make_train function
    
    test_set - The test set constucted earlier from make_train function
    
    
    
    returns:
    
    The mean AUC (area under the Receiver Operator Characteristic curve) of the test set only on user-item interactions
    there were originally zero to test ranking ability in addition to the most popular items as a benchmark.
    '''
    
    
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = [] # To store popular AUC scores
    # Spocita cetnost items podle useru kolikrat se vyskytuje
    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1) # Get sum of item iteractions to find most popular
    item_vecs = predictions[1]
    for user in altered_users: # Iterate through each user that had an item altered
        training_row = training_set[user,:].toarray().reshape(-1) # Get the training set row
        zero_inds = np.where(training_row == 0) # Find where the interaction had not yet occurred
        # Get the predicted values based on our user/item vectors
        user_vec = predictions[0][user,:]
        #pred = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1)
        pred = user_vec.dot(item_vecs.T).toarray()[0,zero_inds].reshape(-1)
        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training 
        pop = pop_items[zero_inds] # Get the item popularity for our chosen items
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score
    # End users iteration
    
    score_model = np.mean(store_auc)
    score_base = np.mean(popularity_auc)
    print('score model: ', score_model, '/ score_base : ', score_base )

    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))  
   # Return the mean AUC rounded to three decimal places for both test and popularity benchmark


def get_items_purchased(siebel_id, mf_train, customers_list, products_list, item_lookup):
    '''
    This just tells me which items have been already purchased by a specific user in the training set. 
    
    parameters: 
    
    customer_id - Input the customer's id number that you want to see prior purchases of at least once
    
    mf_train - The initial ratings training set used (without weights applied)
    
    customers_list - The array of customers used in the ratings matrix
    
    products_list - The array of products used in the ratings matrix
    
    item_lookup - A simple pandas dataframe of the unique product ID/product descriptions available
    
    returns:
    
    A list of item IDs and item descriptions for a particular customer that were already purchased in the training set
    '''
    cust_ind = np.where(customers_list == customer_id)[0][0] # Returns the index row of our customer id
    purchased_ind = mf_train[cust_ind,:].nonzero()[1] # Get column indices of purchased items
    prod_codes = products_list[purchased_ind] # Get the stock codes for our purchased items
    return item_lookup.loc[item_lookup.prod_code.isin(prod_codes)]

def rec_items(siebel_id, mf_train, user_vecs, item_vecs, customer_list, item_list, item_lookup, num_items = 10):
    '''
    This function will return the top recommended items to our users 
    
    parameters:
    
    customer_id - Input the customer's id number that you want to get recommendations for
    
    mf_train - The training matrix you used for matrix factorization fitting
    
    user_vecs - the user vectors from your fitted matrix factorization
    
    item_vecs - the item vectors from your fitted matrix factorization
    
    customer_list - an array of the customer's ID numbers that make up the rows of your ratings matrix 
                    (in order of matrix)
    
    item_list - an array of the products that make up the columns of your ratings matrix
                    (in order of matrix)
    
    item_lookup - A simple pandas dataframe of the unique product ID/product descriptions available
    
    num_items - The number of items you want to recommend in order of best recommendations. Default is 10. 
    
    returns:
    
    - The top n recommendations chosen based on the user/item vectors for items never interacted with/purchased
    '''
    
    cust_ind = np.where(customer_list == siebel_id)[0][0] # Returns the index row of our customer id
    pref_vec = mf_train[cust_ind,:].toarray() # Get the ratings from the training set ratings matrix
    pref_vec = pref_vec.reshape(-1) + 1 # Add 1 to everything, so that items not purchased yet become equal to 1
    pref_vec[pref_vec > 1] = 0 # Make everything already purchased zero
    rec_vector = user_vecs[cust_ind,:].dot(item_vecs) # Get dot product of user vector and all item vectors
    # Scale this recommendation vector between 0 and 1
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.T)[:,0] 
    #recommend_vector = pref_vec*rec_vector_scaled 
    pref_vec = pref_vec > 0
    recommend_vector = rec_vector_scaled[pref_vec]

    rec_df = pd.DataFrame(recommend_vector.toarray())
    rec_df.index.name = 'prod_id'
    rec_df['siebel_id'] = siebel_id
    rec_df.columns = ['rec_score', 'siebel_id']
    rec_df.to_sql('recommendations', oracle_con, if_exists="append")

    # # Items already purchased have their recommendation multiplied by zero
    # #product_idx = np.argsort(recommend_vector)[::-1][:num_items] # Sort the indices of the items into order 
    # product_idx = np.argsort(recommend_vector.toarray()[:, 0])[::1][:num_items]
    # # of best recommendations
    # rec_list = [] # start empty list to store items
    # for index in product_idx:
    #     code = item_list[index]
    #     rec_list.append([code, item_lookup.prod_desc.loc[item_lookup.prod_code == code].iloc[0]]) 
    #     # Append our descriptions to the list
    # codes = [item[0] for item in rec_list]
    # descriptions = [item[1] for item in rec_list]
    # final_frame = pd.DataFrame({'prod_code': codes, 'prod_desc': descriptions}) # Create a dataframe 
    # return final_frame[['prod_code', 'prod_desc']] # Switch order of columns around

# Main flow

retail_data = data_extraction(oracle_con)

grouped_purchased, item_lookup = data_cleaning(retail_data)

purchases_sparse, customers, products = create_matrix(grouped_purchased)

product_train, product_test, product_users_altered = make_train(purchases_sparse, pct_test = 0.2)

user_vecs, item_vecs = implicit_weighted_als(product_train, lambda_val = 0.1, alpha = 4%, iterations = 50, rank_size=20, seed=0)

score_model, score_base = calc_mean_auc(product_train, product_users_altered, [sparse.csr_matrix(user_vecs), sparce.csr_matrix(item_vec.T)], product_test)

customers_arr = np.array(customers)
products_arr = np.array(products)

query = "delete from recommendations"
oracle_con.execute(query)

for c in custommers_arr:
    rec_items(c, product_train, user_vecs, item_vecs, customers_arr, products_arr, item_lookup, num_items= 10)

# print(customers_arr[:5])
# print(get_items_purchased(1122010, product_train, customers_arr, products_arr, item_lookup))

# recommendations = rec_items(1122010, product_train, user_vecs, item_vecs, customers_arr, products_arr, item_lookup, num_items = 10)
# print(recommendations)