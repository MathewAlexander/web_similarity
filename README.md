# web_similarity
A tool to calculate the similarity between the contents of two websites.

## To run the project locally

cd into the project folder and execute the below command from the terminal

```
pip install -r requirements.txt
```

execute the following command to call the run.py from the project directory and paste the urls when prompted

```
python run.py
```


## To run the project on colab
Please refer to the notebook  [Web_similarity.ipynb](https://github.com/MathewAlexander/web_similarity/blob/main/Web_similarity.ipynb) which can be exected directly on colab
or click [here](https://colab.research.google.com/drive/1TKIZZOIRESlyEC4RQXRxa-5ofUQG32Y0?usp=sharing)


## To check the output

The out put will be displayed on the terminal and also will get written into a log file(app.log)


## Sample of supported websites
Stories on skynews  eg.https://news.sky.com/story/boris-johnson-the-us-is-our-closest-and-most-important-ally-12127283

Stories on nytimes  eg.https://www.nytimes.com/2020/11/08/us/politics/georgia-politics.html

Stories on reuters  eg.https://in.reuters.com/article/us-usa-election-trade/bidens-trade-policy-will-take-aim-at-china-embrace-allies-idINKBN27N0W4

Storis on ndtv      eg.https://www.ndtv.com/india-news/details-of-dissent-letter-to-sonia-gandhi-steady-decline-no-honest-inspection-2286399


## Algorithms

1.Web scrapping        :beautifulsoup

2.Web content cleaning : Clustering with sentence embeddings

3.content similarity  : A naive implementation of Sentence Mover's Distance with Sentence embeddings

