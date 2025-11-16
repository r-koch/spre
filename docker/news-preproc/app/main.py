from news_preproc import generate_lagged_features

def lambda_handler(event=None, context=None):
    return generate_lagged_features(context)
