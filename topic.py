import pandas as pd

wine = pd.read_csv('winemag-data-130k-v2.csv')

wine = wine.sort_values('points', ascending= False)

wine = wine[wine['points'] > 89]

wine.isnull().sum()

#wine = wine.dropna()
#var = input()

#wine = wine[wine['variety'] == var]


pop = wine[['country','province','variety','points','description','region_1']]

from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5000)

X = count.fit_transform(pop['description'].values)


from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch')
X_topics = lda.fit_transform(X)



n_top_words = 5
feature_names = count.get_feature_names()

for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort()\
                        [:-n_top_words - 1:-1]]))

print("please select Topic")

suuti = input()

good = X_topics[:, int(suuti)-1].argsort()[::-1]

for iter_idx, wine_idx in enumerate(good[:5]):
  hero = pop.iloc[wine_idx]
  print('\ngood wine #%d:' % (iter_idx + 1))
  print(hero)