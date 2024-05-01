import asyncio

loop = asyncio.get_event_loop()
if loop.is_closed():
    asyncio.set_event_loop(asyncio.new_event_loop())

import time
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from db import insert_data
import sqlite3
from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel, GPT2Model, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import numpy as np
from typing import Optional
import requests
from bs4 import BeautifulSoup


def get_bert_embedding(text, model_name='bert-base-uncased'):
    start_time = time.time()
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    end_time = time.time()
    return embedding, end_time - start_time

def get_gpt_embedding(text, model_name='gpt2'):
    start_time = time.time()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = GPT2Model.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    end_time = time.time()
    return embedding, end_time - start_time

def cosine_similarity_score(embedding1, embedding2):
    start_time = time.time()
    score = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
    end_time = time.time()
    return score, end_time - start_time

class ActionRecommendDomain(Action):
    def name(self):
        return "action_recommend_domain"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        fav_module = tracker.get_slot('fav_module_slot')
        least_fav_module = tracker.get_slot('least_fav_module_slot')
        interested_domain = tracker.get_slot('interested_domain_slot')
        not_interested_domain = tracker.get_slot('not_interested_domain_slot')

        domains = ["Machine Learning", "Web Development", "Cloud Computing", "Software Development", "Deep Learning", "Artificial Intelligence", "Data Science","Machine Learning","Computer Networks","Blockchain Technology","Cyber Security","Network Security"]
        if not_interested_domain in domains:
            domains.remove(not_interested_domain)

        fav_module_emb_bert, bert_time = get_bert_embedding(fav_module)
        fav_module_emb_gpt, gpt_time = get_gpt_embedding(fav_module)
        int_domain_emb_bert, _ = get_bert_embedding(interested_domain)
        int_domain_emb_gpt, _ = get_gpt_embedding(interested_domain)

        bert_total_time = bert_time
        gpt_total_time = gpt_time
        scores = {}

        for domain in domains:
            domain_emb_bert, bert_time = get_bert_embedding(domain)
            domain_emb_gpt, gpt_time = get_gpt_embedding(domain)
            bert_total_time += bert_time
            gpt_total_time += gpt_time

            score_bert, time_bert = cosine_similarity_score(fav_module_emb_bert, domain_emb_bert)
            score_gpt, time_gpt = cosine_similarity_score(fav_module_emb_gpt, domain_emb_gpt)
            score_int_bert, time_int_bert = cosine_similarity_score(int_domain_emb_bert, domain_emb_bert)
            score_int_gpt, time_int_gpt = cosine_similarity_score(int_domain_emb_gpt, domain_emb_gpt)

            score = (score_bert + score_gpt + score_int_bert + score_int_gpt) / 4
            bert_total_time += time_bert + time_int_bert
            gpt_total_time += time_gpt + time_int_gpt
            scores[domain] = score

        recommended_domain = max(scores, key=scores.get)
        dispatcher.utter_message(text=f"BERT processing time: {bert_total_time:.2f} seconds, GPT processing time: {gpt_total_time:.2f} seconds.")

        return [SlotSet('rec_dom_slot',recommended_domain)]




class Actionout(Action):

    def name(self) -> Text:
        return "action_student_interested_domain"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        conn = sqlite3.connect('studentdemo.db')
        cursor = conn.cursor()
        cursor.execute('''SELECT lecture_1, lecture_2, lecture_3 FROM lecture_test''')
        lecture_details = cursor.fetchone()
        f = tracker.get_slot('rec_dom_slot')
        if lecture_details:
            lecture_1, lecture_2, lecture_3 = lecture_details
            dispatcher.utter_message(
                text=f"For a project that intersects {f}, you have some excellent Lectures at our university. \nLecture details: \nLecture 1: {lecture_1}\nLecture 2: {lecture_2}\nLecture 3: {lecture_3}\n")
        else:
            dispatcher.utter_message(text="No lecture details found in the database.")

        conn.close()

        return []

class Actionout(Action):

    def name(self) -> Text:
        return "action_lecture_research_paper"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        conn = sqlite3.connect('studentdemo.db')
        cursor = conn.cursor()
        cursor.execute('''SELECT lec_1_res, lec_2_res, lec_3_res FROM lecture_research_test''')
        lecture_details = cursor.fetchone()

        if lecture_details:
            lecture_1, lecture_2, lecture_3 = lecture_details
            dispatcher.utter_message(
                text=f"Here you go! \nLecture Research details: \n{lecture_1}\n{lecture_2}\n{lecture_3}\n")
        else:
            dispatcher.utter_message(text="No lecture research details found in the database.")

        conn.close()

        return []




class ActionFav(Action):

    def name(self) -> Text:
        return "action_favourite_module"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        e = next(tracker.get_latest_entity_values('module_name_1'), None)
        l=[]
        if e:
            l.append(SlotSet('fav_module_slot',e))
        return l
class ActionLeastFav(Action):

    def name(self) -> Text:
        return "action_least_favourite_module"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        e = next(tracker.get_latest_entity_values('module_name_2'), None)
        l=[]
        if e:
            l.append(SlotSet('least_fav_module_slot',e))

        return l
class Actionintdom(Action):

    def name(self) -> Text:
        return "action_interested_domain"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        e = next(tracker.get_latest_entity_values('domain_name_1'), None)
        l=[]
        if e:
            l.append(SlotSet('interested_domain_slot',e))
        return l
class Actionnotintdom(Action):

    def name(self) -> Text:
        return "action_not_interested_domain"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        e = next(tracker.get_latest_entity_values('domain_name_2'), None)
        l=[]
        if e:
            l.append(SlotSet('not_interested_domain_slot',e))
        insert_data(tracker.get_slot('cn'), tracker.get_slot('dn'), tracker.get_slot('en'), tracker.get_slot('fn'), tracker.get_slot('fav_module_slot'), tracker.get_slot('least_fav_module_slot'), tracker.get_slot('interested_domain_slot'), tracker.get_slot('not_interested_domain_slot'))

        return l


class ActionFetchAndDisplayLectures(Action):
    def name(self):
        return "action_fetch_and_display_lectures"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]):
        domain_name = tracker.get_slot('rec_dom_slot')
        conn = sqlite3.connect('studentdemo.db')
        cursor = conn.cursor()
        query = "SELECT name FROM lecture WHERE lower(domain) = lower(?)"
        cursor.execute(query, (domain_name,))
        lectures = cursor.fetchall()
        conn.close()
        if lectures:
            lecture_names = [name[0] for name in lectures]
            lecture_details = "\n".join(lecture_names)
            return [SlotSet("supervisor_details_slot", lecture_details)]
        else:
            return [SlotSet("supervisor_details_slot", f"No Supervisors found in the domain of {domain_name}.")]



class ActionFetchAndDisplayLecturesResearch(Action):
    def name(self):
        return "action_fetch_and_display_lectures_research"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict]:
        domain_name = tracker.get_slot('rec_dom_slot')  

        conn = sqlite3.connect('studentdemo.db')
        cursor = conn.cursor()

        query = "SELECT research FROM lecture WHERE lower(domain) = lower(?)"
        cursor.execute(query, (domain_name,))
        lectures = cursor.fetchall()
        conn.close()

        if lectures:
            research_details = [research[0] for research in lectures]
            research_output = "\n".join(research_details)
            return [SlotSet("supervisor_research_details_slot", research_output)]
        else:
            return [SlotSet("supervisor_research_details_slot", f"No Supervisors Research Work found in the domain of {domain_name}.")]



class ActionScrapePersonBio(Action):

    def name(self) -> Text:
        return "action_scrape_person_bio"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        person_name_entity = next(tracker.get_latest_entity_values("person"), None)
        if person_name_entity:
            search_name = re.match(r"(\w+)", person_name_entity).group(0).lower()

            valid_names = {
                "yanchao": "yanchao-yu",
                "gordon": "gordon-russell",
                "craig": "craig-thomson",
                "amjad": "amjad-ullah",
                "berk": "berk-canberk",
                "amir": "amir-hussain",
            }

            url_name = valid_names.get(search_name)

            if url_name:
                url = f"https://www.napier.ac.uk/people/{url_name}"

                try:
                    page = requests.get(url)
                    soup = BeautifulSoup(page.content, "html.parser")
                    content = soup.find("div", id="tab-1").find("p").text
                    dispatcher.utter_message(text=content)
                except Exception as e:
                    dispatcher.utter_message(text=f"Could not retrieve information for {person_name_entity}.")
            else:
                dispatcher.utter_message(text=f"Sorry, I don't have information on {person_name_entity}.")
        else:
            dispatcher.utter_message(text="Sorry, I couldn't find the name you're asking about.")

        return []

class ActionTestingnames(Action):

    def name(self) -> Text:
        return "action_testing_names"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        e = next(tracker.get_latest_entity_values('domain_name'), None)
        l=[]
        a= tracker.get_slot('fav_module_slot')
        b= tracker.get_slot('least_fav_module_slot')
        c= tracker.get_slot('interested_domain_slot')
        d= tracker.get_slot('not_interested_domain_slot')
        dispatcher.utter_message(text=f"Your input domains, {a}, {b}, {c},{d} .")

        return l
