

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize 
from bs4 import BeautifulSoup
from bs4.element import Comment
import requests
import urllib.request
from web_parser import *
from googlesearch import search
from bs4 import BeautifulSoup
from bs4.element import Comment
import requests
import urllib.request
import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import wiki_Sum
from wiki_Sum import wiki_Sum
import web_scraper
from web_scraper import extract_text_from_single_web_page
from lsa_summarizer import LsaSummarizer
from sentence_transformers import SentenceTransformer, util
from diophila import OpenAlex
import time

model = SentenceTransformer('all-MiniLM-L6-v2')


nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)



import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()


stopWords = set(stopwords.words("english"))

def getStandard(query):
    page = requests.get("https://en.wikipedia.org/wiki/" + query)
    txt = text_from_html(page.content)
    return txt

def getEmbeddingScores(one, two):
    embeddings1 = model.encode(one, convert_to_tensor=True)
    embeddings2 = model.encode(two, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)    
    return cosine_scores



#returns set of websites after google searching query
def getWebsites(query):
    websites = set()
    for j in search(query, tld="co.in", num=10, stop=10, pause=10):
        if "https://en.wikipedia.org/" not in j:    
            websites.add(j)
    return websites

def getWebsiteInfo(website):
    page  = requests.get(website)
    txt = text_from_html(page.content)
    return txt

def getWebsiteTitle(website):
    page = requests.get(website)
    txt = title_from_html(page.content)
    return txt

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    page_body = soup.body
    texts = page_body.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)


def title_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    text = ""
    for title in soup.find_all('title'):
        text += title.get_text()
    return text


def getEntities(txt):
    doc = nlp(txt)
    return([(X.text, X.label_) for X in doc.ents])


def getMostCommonEntity(entities):
    freq = dict()
    maxCount = 0
    global maxVal
    for entity in entities:
        if entity[1] == "CARDINAL" or entity[1] == "DATE" or entity[1] == "ORDINAL" or entity[1] == "LOC":
            continue
        if entity in freq:
            freq[entity] += 1
        else:
            freq[entity] = 1
    for x in freq:
        if freq[x] > maxCount:
            maxCount = freq[x]
            maxVal = x
    return maxVal


def validateEntity(entity): 
   

    websites = getWebsites(entity)
    entites = []
    data = {}
    maxSim = []
    for url in websites:
        resp = requests.get(url)
        
        # 2. If the response content is 200 - Status Ok, Save The HTML Content:
        if resp.status_code == 200:
            data[url] = resp.text
        text_content = extract_text_from_single_web_page(url)

        if text_content:
            temp = getEntities(str(text_content))

            mostCommon = getMostCommonEntity(temp)

            entites.append(mostCommon)

            maxSim.append([getEmbeddingScores(mostCommon, entity[1:-1]), mostCommon])
    for ten in maxSim:
        val = ten[0]
        if val[0][0] or val[0][1] == 1: 
            return True
    
    return False