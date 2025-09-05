"""News processing module for the Travel Agent application.

This module handles news discovery, content extraction, and dialogue generation
using Google News API and OpenAI services. It provides functionality to:
- Find relevant news topics based on user interests
- Fetch news articles from Google News
- Extract full article content using OpenAI web search
- Generate engaging dialogues from news content
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from GoogleNews import GoogleNews
from openai import OpenAI
from pydantic import BaseModel

from utils import total_token


# Configure logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables and initialize OpenAI client
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Google News client
googlenews = GoogleNews(lang='en', region='US')
googlenews.enableException(True)

# Global variables for Google News topics
GOOGLE_NEWS_TOPIC_MAP: Dict[str, str] = {}
GOOGLE_NEWS_TOPICS: List[str] = []

MAX_TOPICS = 3

# Load Google News topic mappings
try:
    with open("./utils/googlenews_data.json", "r") as f:
        data = json.load(f)
        for topic_name, topic_data in data.items():
            GOOGLE_NEWS_TOPIC_MAP[topic_name] = topic_data["Topic ID"]
            GOOGLE_NEWS_TOPICS.append(topic_name)
        logging.info(f"Loaded {len(GOOGLE_NEWS_TOPICS)} Google news topics")
        logging.info(f"Available topics: {GOOGLE_NEWS_TOPICS}")
except FileNotFoundError:
    logging.error("googlenews_data.json not found. Please ensure the file exists.")
    raise
except json.JSONDecodeError as e:
    logging.error(f"Error parsing googlenews_data.json: {e}")
    raise

class Interest(BaseModel):
    """Pydantic model for user interest topics.
    
    Attributes:
        interest_topic: List of topics that match user interests
    """
    interest_topic: List[str]

def find_users_interest_topic(self_introduction: str) -> List[str]:
    """Find relevant news topics based on user's self-introduction.
    
    Uses OpenAI to analyze the user's introduction and match it with available
    Google News topics. Returns up to 3 most relevant topics.
    
    Args:
        self_introduction: User's description of their interests
        
    Returns:
        List of topic names that match user interests
        
    Raises:
        Exception: If OpenAI API call fails
    """
    system_prompt = f"""
        You are a news assistant that finds, find the topics user will be interested in based on user introduction.
        If the topic is not in the list, find a similar topic in the list.
        Return a maximum of {MAX_TOPICS} topics.
        The topics must exist in the following list: {GOOGLE_NEWS_TOPICS}
    """
    prompt = f"""
        User introduction: {self_introduction}
    """
    
    try:
        response = client.chat.completions.parse(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format=Interest,
        )
        
        # Track token usage
        total_token.TOTAL_INPUT_TOKEN += response.usage.prompt_tokens
        total_token.TOTAL_OUTPUT_TOKEN += response.usage.completion_tokens
        
        topics = response.choices[0].message.parsed.interest_topic
        logging.info(f"Found {len(topics)} matching topics: {topics}")
        return topics
        
    except Exception as e:
        logging.error(f"Error finding user interest topics: {e}")
        raise



class NewsDialog(BaseModel):
    """Pydantic model for news dialogue generation.
    
    Attributes:
        dialog: List of dialogue lines between questioner and answerer
        questioner_tone_style: Description of questioner's speaking style
        answerer_tone_style: Description of answerer's speaking style
    """
    dialog: List[str]
    questioner_tone_style: str
    answerer_tone_style: str

def parse_news_dialogue(article_content: str) -> NewsDialog:
    """Convert news article into engaging dialogue format.
    
    Takes a news article and transforms it into an entertaining conversation
    between two fictional characters using various creative styles and tones.
    
    Args:
        article_content: The full text content of the news article
        
    Returns:
        NewsDialog object containing the dialogue and tone descriptions
        
    Raises:
        Exception: If OpenAI API call fails
    """
    system_prompt = """ 
        You are a creative news storyteller who turns plain news articles into bold, high-energy dialogues designed to grab attention and spark curiosity.

        🔥 Mission:
        Take a news article and transform it into a **fun, exaggerated, and attention-grabbing** conversation between two fictional characters:
        - One person is reacting to the news with excitement, confusion, sarcasm, or curiosity.
        - The other provides colorful, vivid explanations, background, and spicy takes.

        🎭 Style & Tone:
        Each time, randomly pick (or combine) one or more of the following **engaging styles** and define the tone of the questioner and answerer.
        - Dramatic & cinematic
        - Witty & sarcastic
        - Over-the-top excitement
        - Apocalyptic or sci-fi tone
        - Pop culture references (if relevant)
        - Hyper-relatable “meme talk”
        - Internet/forum-style ranting
        - Conspiracy-theorist energy
        - Satirical corporate-speak
        ...or make up something unique. Be creative.
        
        Be bold with tone, metaphors, and comparisons. Surprise the reader. Even if the news is technical or dry, make it **pop**.

        🧠 Structure:
        - Start with a hook or surprising reaction.
        - Then go into a back-and-forth Q&A format.
        - Build intrigue, tension, or humor.
        - End with a memorable punchline or reflection.

        📦 Output Format:
        Return valid JSON strictly in this format:
        {
            "dialog": [
                "First line of dialogue (usually a reaction or question)",
                "Reply with explanation or insight in engaging style",
                "Follow-up question or comment",
                "Next reply",
                "...",
                "Final line (wrap-up, punchline, or reflection)"
            ]
            "questioner_tone_style": "Describe how the questioner speaks (e.g., 'Speaks with wild-eyed curiosity and Gen-Z slang')",
            "answerer_tone_style": "Describe how the answerer speaks (e.g., 'Speaks like a smug but brilliant sci-fi narrator')",
        }
        💡 Tips:
        - Use vivid language, exaggeration, or emotion to make things stick.
        - Vary sentence lengths and tone for rhythm.
        - Keep it human and entertaining.
        - Avoid narration. Only include dialogue lines.
    """

    try:
        response = client.chat.completions.parse(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": article_content}
            ],
            response_format=NewsDialog
        )
        
        # Track token usage
        total_token.TOTAL_INPUT_TOKEN += response.usage.prompt_tokens
        total_token.TOTAL_OUTPUT_TOKEN += response.usage.completion_tokens
        
        dialogue = response.choices[0].message.parsed
        logging.info(f"Generated dialogue with {len(dialogue.dialog)} lines")
        return dialogue
        
    except Exception as e:
        logging.error(f"Error generating news dialogue: {e}")
        raise


def _pick_article(articles: List[Dict[str, Any]]) -> int:
    """Select an article from the list of available articles.
    
    Currently returns the first article (index 0) for consistency.
    Future enhancement: Use AI to select the most relevant article.
    
    Args:
        articles: List of article dictionaries from Google News
        
    Returns:
        Index of the selected article
        
    Note:
        Could be enhanced to use random selection or AI-based selection
    """
    if not articles:
        raise ValueError("No articles provided")
    
    # For now, always pick the first article for consistency
    # TODO: Implement AI-based article selection
    return 0
    

def _get_article(article_info: Dict[str, Any]) -> str:
    """Fetch the full article content using OpenAI web search.
    
    Uses OpenAI's web search capability to find and extract the complete
    article content based on the article metadata from Google News.
    
    Args:
        article_info: Dictionary containing article metadata (title, reporter, media, datetime)
        
    Returns:
        Full text content of the article
        
    Raises:
        Exception: If web search or content extraction fails
    """
    prompt = f"""
        Search for the following article and return the original article content.
        Only return the article text, do not add extra commentary.
        
        Title: {article_info.get("title", "N/A")}
        Reporter: {article_info.get("reporter", "N/A")}
        Media: {article_info.get("media", "N/A")}
        Datetime: {article_info.get("datetime", "N/A")}
    """
    
    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=prompt,
            tools=[{"type": "web_search"}],
        )
        
        logging.info(f"Successfully fetched article content for: {article_info.get('title', 'Unknown')}")
        
        # Track token usage
        total_token.TOTAL_INPUT_TOKEN += response.usage.input_tokens
        total_token.TOTAL_OUTPUT_TOKEN += response.usage.output_tokens
        
        return response.output_text
        
    except Exception as e:
        logging.error(f"Error fetching article content: {e}")
        raise

def get_news(topics: List[str]) -> List[Dict[str, Any]]:
    """Fetch news articles for the given topics.
    
    For each topic, fetches news from Google News, selects an article,
    and retrieves the full content using web search.
    
    Args:
        topics: List of topic names to fetch news for
        
    Returns:
        List of dictionaries containing article information and full content
        
    Raises:
        KeyError: If topic is not found in GOOGLE_NEWS_TOPIC_MAP
        Exception: If news fetching or content extraction fails
    """
    news_articles = []
    
    for topic in topics:
        try:
            if topic not in GOOGLE_NEWS_TOPIC_MAP:
                logging.warning(f"Topic '{topic}' not found in topic map. Skipping.")
                continue
                
            # Set topic and fetch news
            googlenews.set_topic(GOOGLE_NEWS_TOPIC_MAP[topic])
            googlenews.get_news()
            results = googlenews.results()
            
            if not results:
                logging.warning(f"No news found for topic: {topic}")
                googlenews.clear()
                continue
                
            logging.info(f"Found {len(results)} articles for topic: {topic}")
            
            # Select and process article
            article_index = _pick_article(results)
            article_info = results[article_index]
            
            # Clear previous results to avoid conflicts
            googlenews.clear()
            
            # Fetch full article content
            full_article = _get_article(article_info)
            article_info["article"] = full_article
            
            news_articles.append(article_info)
            logging.info(f"Successfully processed article: {article_info.get('title', 'Unknown')}")
            
        except Exception as e:
            logging.error(f"Error processing topic '{topic}': {e}")
            googlenews.clear()  # Ensure cleanup even on error
            continue
    
    logging.info(f"Successfully fetched {len(news_articles)} news articles")
    return news_articles
            
