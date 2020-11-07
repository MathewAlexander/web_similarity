
from model import USE,SBERT,Scrap



#Initializing the model
model=SBERT()



print('Provide link 1')
link_1=str(input())
content_1=Scrap.get_web_content(link_1)
web_content_1,noise=model.cluster(content_1)
print('Web content :\n {}'.format(web_content_1))
print('filtered noise :\n {}'.format(noise))

print('Provide link 2')
link_2=str(input())
content_2=Scrap.get_web_content(link_2)
noise,web_content_2=model.cluster(content_2)
print('Web content :\n {}'.format(web_content_2))
print('filtered noise :\n {}'.format(noise))


score=model.one_to_one(web_content_1,web_content_2)

print(score)
