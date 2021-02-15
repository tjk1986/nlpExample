from regex import sub

def parser(x):
    x = x.lower()
    x = sub(r'((\[.*\]\(.*\)))', '', x) # remove markdown urls
    x = sub(r'\b(http|www.)\S*', '', x) # remove urls
    x = sub(r'[^A-Za-z ]', '', x) # remove special characters
    x = ' '.join(x.strip().split()) # remove extra with spaces
    x = x.split(' ')

    return x
