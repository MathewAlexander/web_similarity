
import logging
import os
from model import USE,SBERT,Scrap

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='app.log', filemode='w',format='%(message)s', level=logging.CRITICAL)

def printAndLogInfo(customMessage,exceptionMessage=None):
    print(customMessage)
    try:
        logging.critical(customMessage)
    except Exception as err:
        logging.critical('error while logging : {}'.format(str(err)))

    if exceptionMessage:
        print(str(exceptionMessage))
        logging.critical(exceptionMessage)

# Function to print & log errors
def printAndLogError(customMessage,exceptionMessage=None):
    print('ERROR!!! ' +customMessage)
    logging.critical(customMessage)
    if exceptionMessage:
        print(str(exceptionMessage))
        logging.critical(exceptionMessage)
    time.sleep(10)







#Initializing the model
model=SBERT()



while True:


    link_1=str(input('Paste the first url...'))
    content_1=Scrap.get_web_content(link_1)
    web_content_1,noise=model.cluster(content_1)
    printAndLogInfo('\n\n\n\t\t###### Printing the Web content ######\n')
    printAndLogInfo(web_content_1)
    printAndLogInfo('\t\t######Printing the noise filtered out ######\n')
    printAndLogInfo(noise)

    link_2=str(input('Paste the second url..'))
    content_2=Scrap.get_web_content(link_2)
    web_content_2,noise=model.cluster(content_2)
    printAndLogInfo('\n\n\n\\t\t###### Printing the Web content ######\n')
    printAndLogInfo(web_content_2)
    printAndLogInfo('\t\t######Printing the noise filtered out ######\n')
    printAndLogInfo(noise)


    score_smd=model.sentence_movers_distance(web_content_1,web_content_2)*100
    score_one_to_one=model.one_to_one(web_content_1,web_content_2)*100
    score=(score_smd+score_one_to_one)/2

    printAndLogInfo("\n\n\nThe percentage of similarity is  : {}".format(score))

    next_action=input('\n\n\nCheck another set of URLs ? y/n \n')
    if next_action=='n':
        break
