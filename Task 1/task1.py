import gzip
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Open and load the data
json_data = gzip.open('goemotions.json.gz', 'r')
raw_data = json.load(json_data)

# Extract posts, emotions and sentiments
posts = [x[0] for x in raw_data]
emotions = [x[1] for x in raw_data]
sentiments = [x[2] for x in raw_data]

counted_emotions = Counter(emotions)
counted_sentiments = Counter(sentiments)

# Plot the distributions

plt.pie(counted_emotions.values(), labels = counted_emotions.keys())
plt.show()

plt.pie(counted_sentiments.values(), labels = counted_sentiments.keys())
plt.show()