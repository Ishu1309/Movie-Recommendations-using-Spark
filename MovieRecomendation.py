import sys
from pyspark import SparkContext

def check_usage():
    """
    Check the usage movie recommendations
    Raise an exception in case of missing arguments.
    file1 - format movieid, moviename, ...
    file2 - format userid, movieid, ratings
    :return: None
    """
    if len(sys.argv) != 3:
        raise Exception('Usage:spark-submit MovieRecomendation.py <file1> <file2>')


def extract_tuple_from_ratings_data(line):
    """
    Extract a rating record from movie ratings file in which data
    is separated using tab ['\t']
    :param str line:
    :return:
    """
    fields = line.strip().split('\t')
    return (int(fields[0]), int(fields[1]), int(fields[2]))


def extract_movie_file_name(line):
    """
    Extract movie id and movie name
    :param line:
    :return:
    """
    fields = line.strip().split('|')
    return (int(fields[0]), fields[1])


def compute_input_for_similarity_metrics(data):
    """
    Compute inputs to similarity metrics,
    product of ratings, rating of both movies, square of ratings
    :param data:
    :return:
    """
    key = (data[1][0][1], data[1][1][1])
    stats = (data[1][0][2] * data[1][1][2], data[1][0][2], data[1][1][2], data[1][0][2] * data[1][0][2],
             data[1][1][2] * data[1][1][2], data[1][0][3], data[1][1][3])
    return (key, stats)


def gather_similarity_metrics(data):
    """
    Gather the computed similarity metrics
    :param data:
    :return:
    """
    key = data[0]
    vals = data[1]
    size = len(vals)
    dot_product = 0
    rating_sum = 0
    rating_sum2 = 0
    rating_sq = 0
    rating_sq2 = 0
    numraters = []
    numraters2 = []
    for i in range(size):
        dot_product = dot_product + vals[i][0]
        rating_sum = rating_sum + vals[i][1]
        rating_sum2 = rating_sum2 + vals[i][2]
        rating_sq = rating_sq + vals[i][3]
        rating_sq2 = rating_sq2 + vals[i][4]
        numraters.append(vals[i][5])
        numraters2.append(vals[i][6])
    max_numraters = max(numraters)
    max_numraters2 = max(numraters2)
    return (key, (size, dot_product, rating_sum, rating_sum2, rating_sq, rating_sq2, max_numraters, max_numraters2))


def similarities(fields):
    """
    Calculate jaccard similarity
    :param fields:
    :return:
    """
    key = fields[0]
    (size, dot_product, rating_sum, rating2_sum, rating_norm_sq, rating2_norm_sq, num_raters, num_raters2) = fields[1]
    deno = (num_raters + num_raters2 - size)
    if deno != 0:
        jaccardCorrelation = size / (num_raters + num_raters2 - size)
    else:
        jaccardCorrelation = -1
    return (key, round(jaccardCorrelation, 4))


if __name__ == '__main__':
    check_usage()

    # extract all command line arguments
    movie_id_name_file = sys.argv[2]
    user_movie_ratings_file = sys.argv[3]

    sc = SparkContext(sys.argv[1], "Movie Recommendations")
    umr_rdd = sc.textFile(user_movie_ratings_file)
    mni_rdd = sc.textFile(movie_id_name_file)

    # Extract user movie and rating's data
    user_movie_rating = umr_rdd.map(extract_tuple_from_ratings_data)
    movie_name = mni_rdd.map(extract_movie_file_name)

    # Extract number of raters per movie, group by movie, joined with number of raters per movie
    # and extracted tuple <user, movie, rating, number of raters>
    num_of_raters_per_movie = user_movie_rating.groupBy(lambda x: x[1]).map(lambda x: (x[0], len(x[1])))
    umr_joined_nrpm = user_movie_rating.groupBy(lambda x: x[1]).map(lambda x: (x[0], list(x[1]))).join(num_of_raters_per_movie)
    user_movie_rating_number_of_raters = umr_joined_nrpm.flatMap(lambda x : list(map(lambda y: (y[0], y[1], y[2], x[1][1]), x[1][0])))

    # get movie name instead of movie id
    movie_name_RDD = movie_name.map(lambda x: (x[0], (x[1]))).join(user_movie_rating_number_of_raters.map(lambda x: (x[1], (x[0], x[2], x[3]))))
    user_movie_rating_number_of_raters = movie_name_RDD.map(lambda x: (x[1][1][0], x[1][0], x[1][1][1], x[1][1][2]))

    # making rdd ready for join in next step
    movie_1 = user_movie_rating_number_of_raters.keyBy(lambda x: x[0])

    # join on user and filter movie pairs so we won't count twice
    movie_pairs = user_movie_rating_number_of_raters.keyBy(lambda x: x[0]).join(movie_1).filter(lambda y: (y[1][0][1] < y[1][1][1]))

    # Gather similarity metrics
    vector_calcs1 = movie_pairs.map(lambda x: compute_input_for_similarity_metrics(x))
    vector_calcs2 = vector_calcs1.groupByKey().map(lambda x: (x[0], list(x[1])))
    vector_calcs = vector_calcs2.map(lambda x: gather_similarity_metrics(x))

    # Calculate Jaccard Similarity
    sim = vector_calcs.map(lambda x: similarities(x))

    #Top - 20 for a movie, example - Star Wars
    sim1 = sim.filter(lambda x: x[0][0] == 'Star Wars (1977)').takeOrdered(20, key=lambda x: -x[1])
    sim1