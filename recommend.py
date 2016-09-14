D"""A Yelp-powered Restaurant Recommendation Program"""

from abstractions import *
from utils import distance, mean, zip, enumerate, sample
from visualize import draw_map
from data import RESTAURANTS, CATEGORIES, USER_FILES, load_user_file
from ucb import main, trace, interact

def find_closest(location, centroids):
    """Return the item in CENTROIDS that is closest to LOCATION. If two
    centroids are equally close, return the first one.

    >>> find_closest([3, 4], [[0, 0], [2, 3], [4, 3], [5, 5]])
    [2, 3]
    """
    "*** YOUR CODE HERE ***"
    return min(centroids, key=lambda x: [distance(location, x)]) 
    # The key is to check the distance between the location and different centroids

def group_by_first(pairs):
    """Return a list of pairs that relates each unique key in [key, value]
    pairs to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_first(example)
    [[2, 3, 2], [2, 1], [4]]
    """
    # Optional: This implementation is slow because it traverses the list of
    #           pairs one time for each key. Can you improve it?
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    return [[y for x, y in pairs if x == key] for key in keys]

def group_by_centroid(restaurants, centroids):
    """Return a list of lists, where each list contains all restaurants nearest
    to some item in CENTROIDS. Each item in RESTAURANTS should appear once in
    the result, along with the other restaurants nearest to the same centroid.
    No empty lists should appear in the result.
    """
    "*** YOUR CODE HERE ***"
    return group_by_first([[find_closest(restaurant_location(restaurant), centroids), restaurant] for restaurant in restaurants])
    # The key in the pair is centroid, which is the result of find_closest, 
    # and the value in the pair is the restaurant
    
    

def find_centroid(restaurants):
    """Return the centroid of the locations of RESTAURANTS."""
    "*** YOUR CODE HERE ***"
    return [mean([restaurant_location(restaurant)[0] for restaurant in restaurants]), mean([restaurant_location(restaurant)[1] for restaurant in restaurants])]
    # First, extract all the latitudes from restaurants and take the mean of all latitudes
    # Then, extract all the longitudes from restaurants and take the mean of all longitudes
    # Finally, make a new list containing the latitude and the longitude of the centroid
    # which are the mean latitudes and longitudes of all restaurants

def k_means(restaurants, k, max_updates=100):
    """Use k-means to group RESTAURANTS by location into K clusters."""
    assert len(restaurants) >= k, 'Not enough restaurants to cluster'
    old_centroids, n = [], 0
    # Select initial centroids randomly by choosing K different restaurants
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]

    while old_centroids != centroids and n < max_updates:
        old_centroids = centroids
        "*** YOUR CODE HERE ***"
        centroids = []
        clusters = group_by_centroid(restaurants, old_centroids)
        for cluster in clusters:
            centroids.append(find_centroid(cluster))
        n += 1
    return centroids
    # Using group_by_centroid function to group restaurants into clusters
    # where each cluster contains all restaurants closest to the same centroid
    # Bind new centroids to a new list of the centroids of all non-empty clusters
    # by using find_centroid function on every cluster in clusters

def find_predictor(user, restaurants, feature_fn):
    """Return a rating predictor (a function from restaurants to ratings),
    for USER by performing least-squares linear regression using FEATURE_FN
    on the items in RESTAURANTS. Also, return the R^2 value of this model.

    Arguments:
    user -- A user
    restaurants -- A sequence of restaurants
    feature_fn -- A function that takes a restaurant and returns a number
    """
    reviews_by_user = {review_restaurant_name(review): review_rating(review)
                       for review in user_reviews(user).values()}

    xs = [feature_fn(r) for r in restaurants]
    ys = [reviews_by_user[restaurant_name(r)] for r in restaurants]

    "*** YOUR CODE HERE ***"
    zip_xy = zip(xs, ys)
    S_xx = 0
    S_yy = 0
    S_xy = 0
    for x in xs:
        S_xx += pow(x - mean(xs), 2)
    for y in ys:
        S_yy += pow(y - mean(ys), 2)
    for pair in zip_xy:
        S_xy += (pair[0] - mean(xs)) * (pair[1] - mean(ys))

    b = S_xy / S_xx
    a = mean(ys) - b * mean(xs)
    r_squared = pow(S_xy, 2) / (S_xx * S_yy)


    def predictor(restaurant):
        return b * feature_fn(restaurant) + a

    return predictor, r_squared
    # First, group xs and ys into one list by zip function
    # Then, define S_xx, S_yy, S_xy according to their definitions
    # Next, define b, a, r_squared according to their definitions
    # Return predictor and r_squared at the end

def best_predictor(user, restaurants, feature_fns):
    """Find the feature within FEATURE_FNS that gives the highest R^2 value
    for predicting ratings by the user; return a predictor using that feature.

    Arguments:
    user -- A user
    restaurants -- A dictionary from restaurant names to restaurants
    feature_fns -- A sequence of functions that each takes a restaurant
    """
    reviewed = list(user_reviewed_restaurants(user, restaurants).values())
    "*** YOUR CODE HERE ***"
    dict_for_preds_and_r2_value = {}
    for feature_fn in feature_fns:
        preds, r2_value = find_predictor(user, reviewed, feature_fn)
        dict_for_preds_and_r2_value[preds] = r2_value
    return max([preds for preds in dict_for_preds_and_r2_value], key = lambda preds: dict_for_preds_and_r2_value[preds])
    # First, create a dictionary that used to store all different predictors and their associated r_square values
    # Then, use find_predictor function to evaluate all the predictor lines and r_square values
    # Last, use max function to return the predictor that has the largest r_squared value

def rate_all(user, restaurants, feature_functions):
    """Return the predicted ratings of RESTAURANTS by USER using the best
    predictor based a function from FEATURE_FUNCTIONS.

    Arguments:
    user -- A user
    restaurants -- A dictionary from restaurant names to restaurants
    """
    # Use the best predictor for the user, learned from *all* restaurants
    # (Note: the name RESTAURANTS is bound to a dictionary of all restaurants)
    predictor = best_predictor(user, RESTAURANTS, feature_functions)
    "*** YOUR CODE HERE ***"
    restaurant_rating_dictionary = {}
    for restaurant_name in restaurants.keys():
        if restaurant_name in user_reviewed_restaurants(user, restaurants):
            restaurant_rating_dictionary[restaurant_name] = user_rating(user, restaurant_name)
        else:
            restaurant_rating_dictionary[restaurant_name] = predictor(restaurants[restaurant_name])
    return restaurant_rating_dictionary
    # First, create an empty dictionary that will be used to store all different restaurants and their associated ratings
    # Then, check if a particular restaurant has been rated by the user already, using user_reviewed_restaurants
    # If yes, set the respective rating to the user rating, using user_rating function
    # If no, set the respective rating to the predicted rating, using predictor function
    # Finally, return the dictionary with different restaurant names binded to different ratings




def search(query, restaurants):
    """Return each restaurant in RESTAURANTS that has QUERY as a category.

    Arguments:
    query -- A string
    restaurants -- A sequence of restaurants
    """
    "*** YOUR CODE HERE ***"
    return [restaurant for restaurant in restaurants if query in restaurant_categories(restaurant)]
    # Check if the query is a category of a restaurant
    # Return a list of restaurants that fulfill the criteria

def feature_set():
    """Return a sequence of feature functions."""
    return [restaurant_mean_rating,
            restaurant_price,
            restaurant_num_ratings,
            lambda r: restaurant_location(r)[0],
            lambda r: restaurant_location(r)[1]]

@main
def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        description='Run Recommendations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--user', type=str, choices=USER_FILES,
                        default='test_user',
                        metavar='USER',
                        help='user file, e.g.\n' +
                        '{{{}}}'.format(','.join(sample(USER_FILES, 3))))
    parser.add_argument('-k', '--k', type=int, help='for k-means')
    parser.add_argument('-q', '--query', choices=CATEGORIES,
                        metavar='QUERY',
                        help='search for restaurants by category e.g.\n'
                        '{{{}}}'.format(','.join(sample(CATEGORIES, 3))))
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict ratings for all restaurants')
    args = parser.parse_args()

    # Select restaurants using a category query
    if args.query:
        results = search(args.query, RESTAURANTS.values())
        restaurants = {restaurant_name(r): r for r in results}
    else:
        restaurants = RESTAURANTS

    # Load a user
    assert args.user, 'A --user is required to draw a map'
    user = load_user_file('{}.dat'.format(args.user))

    # Collect ratings
    if args.predict:
        ratings = rate_all(user, restaurants, feature_set())
    else:
        restaurants = user_reviewed_restaurants(user, restaurants)
        ratings = {name: user_rating(user, name) for name in restaurants}

    # Draw the visualization
    restaurant_list = list(restaurants.values())
    if args.k:
        centroids = k_means(restaurant_list, min(args.k, len(restaurant_list)))
    else:
        centroids = [restaurant_location(r) for r in restaurant_list]
    draw_map(centroids, restaurant_list, ratings)
