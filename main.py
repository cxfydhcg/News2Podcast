import json
from utils.news import get_news, find_users_interest_topic, parse_news_dialogue
from utils.news_to_audio import text_to_speech
from utils import total_token
import time
from datetime import datetime
import logging
import sys

sys.path.append("./utils")
logging.basicConfig(
    filename='app.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    
    introduction = "I’m interested in technology, shopping stores, and real estate."

    user_interest_topics = find_users_interest_topic(introduction)
    logging.info(f"User interest topics: {user_interest_topics}")
    # user_interest_topics = user_interest_topics[:1]
    news = get_news(user_interest_topics)
    logging.info(f"News: {news}")
    for n in news:
        print(n["title"])
        dialog = parse_news_dialogue(n["article"])
        n["dialog"] = dialog.dialog
        logging.info(f"Dialog: {n['dialog']}")
        n["questioner_tone_style"] = dialog.questioner_tone_style
        logging.info(f"Questioner tone style: {n['questioner_tone_style']}")
        n["answerer_tone_style"] = dialog.answerer_tone_style
        logging.info(f"Answerer tone style: {n['answerer_tone_style']}")

        n["datetime"] = n["datetime"].isoformat() if type(n["datetime"]) == datetime else n["datetime"]

    # # Save news list to json file
    with open('news.json', 'w') as f:
        json.dump(news, f, indent=4, ensure_ascii=False)

    # Load news list from json file
    # with open('news.json', 'r') as f:
    #     news = json.load(f)
    # Text to speech
    for n in news:
        file_name = text_to_speech(n["dialog"], n["questioner_tone_style"], n["answerer_tone_style"])
        logging.info(f"File name: {file_name}")
        n["filepath"] = f"speech_files/{file_name}"
        
    with open('news.json', 'w') as f:
        json.dump(news, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    start_time = time.time()
    main()
    logging.info(f"Time taken: {time.time() - start_time} seconds")
    logging.info(f"Total input token: {total_token.TOTAL_INPUT_TOKEN}")
    logging.info(f"Total output token: {total_token.TOTAL_OUTPUT_TOKEN}")
