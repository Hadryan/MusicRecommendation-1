import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

train = pd.read_csv("audio_features_spotify_all.csv")
train_modified = train.drop(['analysis_url','track_href','uri','id','type','song_title'],axis=1)
train_modified = train_modified.dropna(axis=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(train_modified)
pca = PCA(n_components = 2)
principal_components = pca.fit_transform(scaled_data)
model = KMeans(n_clusters=5)
y_train = model.fit_predict(principal_components)
frame = pd.DataFrame(principal_components)
frame['cluster'] = y_train
color = ['blue','green','red','yellow','black']
for k in range(0,5):
  data = frame[frame["cluster"]==k]
  plt.scatter(data[0], data[1], c=color[k])