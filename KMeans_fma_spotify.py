import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

train = pd.read_csv("spotify_fma_merged.csv")
train_modified = train.drop(['track_id','analysis_url','track_href','uri','id','type','track_title','genre','set_size','set_type','album'],axis=1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(train_modified)
pca = PCA(n_components = 2)
principal_components = pca.fit_transform(scaled_data)
model = KMeans(n_clusters=5)
y_train = model.fit_predict(principal_components)
frame = pd.DataFrame(principal_components)
frame['cluster'] = y_train
plt.figure(figsize=(10,10))
color = ['blue','green','red','yellow','cyan']
for k in range(0,5):
  data = frame[frame["cluster"]==k]
  plt.scatter(data[0], data[1], c=color[k],s=400)
for myint in range(0,len(frame)):
  plt.annotate(str(myint+1),(frame[0].tolist()[myint],frame[1].tolist()[myint]))
centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
