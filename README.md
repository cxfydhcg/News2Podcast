# Travel Agent - AI-Powered News to Audio Converter

An intelligent news aggregation and audio conversion system that finds personalized news based on user interests and converts them into engaging audio dialogues using OpenAI's text-to-speech technology.

## Features

- **Personalized News Discovery**: Uses AI to match user interests with relevant Google News topics
- **Intelligent Content Extraction**: Fetches full article content using OpenAI's web search capabilities
- **Dynamic Dialogue Generation**: Converts news articles into engaging conversational format
- **High-Quality Audio Synthesis**: Generates natural-sounding audio using OpenAI's TTS with multiple voice styles
- **Comprehensive Logging**: Tracks all operations with detailed logging for debugging and monitoring

- **Note**: Some sample audio files are provided in the `speech_files/` and sample news data in news.json
- **Cost**: Calculate the cost based on the input and ouput token, and the audio length, or check the usage on offical website

## Project Structure

```
travel_agent/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (create this)
├── news.json             # Generated news data
├── app.log               # Application logs
├── speech_files/         # Generated audio files
    ├── silent.mp3        # file used when combine the audio files
└── utils/
    ├── news.py           # News fetching and processing
    ├── news_to_audio.py  # Text-to-speech conversion
    ├── total_token.py    # Token usage tracking
    └── googlenews_data.json # Google News topic mappings
```

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation and Usage

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**

   Create a `.env` file in the project root:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Customize user interests** in `main.py`:

   ```python
   introduction = "I'm interested in technology, shopping stores, and real estate."
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

### How It Works

1. **Interest Analysis**: The system analyzes your introduction to identify relevant Google News topics
2. **News Aggregation**: Fetches recent news articles from Google News based on identified topics
3. **Content Enhancement**: Uses OpenAI to extract full article content and context
4. **Dialogue Generation**: Converts articles into engaging Q&A format conversations
5. **Audio Synthesis**: Generates high-quality audio files with different voice styles for questioner and answerer
6. **File Management**: Saves all data to `news.json` and audio files to `speech_files/` directory

## Output

- **Audio Files**: Generated in `speech_files/` directory with unique identifiers
- **News Data**: Comprehensive news information saved in `news.json`
- **Logs**: Detailed operation logs in `app.log`

## Configuration

### Supported News Topics

The system supports various Google News categories including:

- Technology
- Business
- Entertainment
- Sports
- Health
- Science
- And many more...

## Dependencies

- `python-dotenv`: Environment variable management
- `pydantic`: Data validation and parsing
- `openai`: OpenAI API integration
- `GoogleNews`: Google News API python library (Thanks for the library)

## Logging

The application provides comprehensive logging including:

- User interest topic identification
- News fetching operations
- Dialogue generation process
- Audio file creation
- Error tracking and debugging information

## Troubleshooting

### Common Issues

1. **Missing OpenAI API Key**

   - Ensure `.env` file exists with valid `OPENAI_API_KEY`

2. **No News Found**

   - Check internet connection
   - Verify user interests match available Google News topics

3. **No Audio Files Generated**

   - Structure output fails when calling OpenAI API

4. **Audio Generation Fails**
   - Ensure `speech_files/` directory exists and silent.mp3 file is in the directory
   - Check OpenAI API quota and permissions

## License

This project is open source. Please check the license file for more details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
Or you can email me at xufengce209@gmail.com
