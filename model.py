

import numpy as np
from numba import jit
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import requests
from bs4 import BeautifulSoup





@jit(nopython=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta



class USE:
  def __init__(self):
    def embed_useT(module):
      with tf.Graph().as_default():
          sentences = tf.placeholder(tf.string)
          embed = hub.Module(module)
          embeddings = embed(sentences)
          session = tf.train.MonitoredSession()
      return lambda x: session.run(embeddings, {sentences: x})

    self.model=embed_useT("./universal-sentence-encoder-lite_2")

  def one_to_one(self,s1, s2):
    embs=self.model([s1,s2])
    score=cosine_similarity_numba(embs[0],embs[1])
    return score
  def cluster(self,content_list):

    corpus_embeddings = self.model(content_list)
    # Then, we perform k-means clustering using sklearn:

    clustering_model = KMeans(n_clusters=2)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    cluster_0=''
    cluster_1=''
    for indx,text in enumerate(content_list):
        if cluster_assignment[indx]==0:
            cluster_0+=text
        if cluster_assignment[indx]==1:
            cluster_1+=text
    if len(cluster_1)>len(cluster_0):
        web_content=cluster_1
        noise=cluster_0
    else:
        web_content=cluster_0
        noise=cluster_1
    return web_content,noise

class SBERT:
  def __init__(self):
    self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

  def one_to_one(self,s1, s2):
    embs=self.model.encode([s1,s2])
    score=cosine_similarity_numba(embs[0],embs[1])
    return score
  def cluster(self,content_list):

    corpus_embeddings = self.model.encode(content_list)
    # Then, we perform k-means clustering using sklearn:

    clustering_model = KMeans(n_clusters=2)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    cluster_0=''
    cluster_1=''
    for indx,text in enumerate(content_list):
        if cluster_assignment[indx]==0:
            cluster_0+=text
        if cluster_assignment[indx]==1:
            cluster_1+=text
    if len(cluster_1)>len(cluster_0):
        web_content=cluster_1
        noise=cluster_0
    else:
        web_content=cluster_0
        noise=cluster_1
    return web_content,noise



class Scrap:

    @staticmethod
    def get_web_content(link):
      article = requests.get(link,verify=False)
      article_content = article.content
      soup = BeautifulSoup(article_content, 'html5lib')
      content_list=[]
      titleTag = soup.html.head.title.text # getting the title
      content_list.append(titleTag)
      for p in soup.findAll('p'):
          content_list.append(p.text)
      return content_list
