from stock_preproc import generate_pivoted_features

def lambda_handler(event=None, context=None):
    return generate_pivoted_features(context)
