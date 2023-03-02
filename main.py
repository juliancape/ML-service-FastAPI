from typing import List

from fastapi import FastAPI, Query
from pydantic import BaseModel
import spacy

nlp_en = spacy.load('en_core_web_sm')
app = FastAPI()
class Article(BaseModel):
    content: str
    comments: List[str] = []

@app.post('/article/')
def recognize_entities(article: Article,
    lang: str = Query(...,min_length = 2, max_length = 2), 
    big_model: bool = Query(False, description = 'Use the big model')):
    
    doc_en = nlp_en(article.content)
    ents = []
    for ent in doc_en.ents:
        ents.append({'text': ent.text, 'label': ent.label_})
    return {'message': article.content, 'ents': ents} 
