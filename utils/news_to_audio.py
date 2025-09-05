
"""Text-to-speech conversion module for news dialogues.

This module provides functionality to convert news dialogues into audio files
using OpenAI's text-to-speech API. It supports multiple speakers with different
voice styles and combines individual audio segments into a single output file.

The module handles:
- Converting dialogue text to speech with speaker-specific voices
- Applying tone and style instructions to each speaker
- Combining multiple audio files with silence intervals
- Managing temporary files and cleanup

Typical usage:
    dialogs = ["Hello there!", "Hi, how are you?"]
    questioner_style = "Speak with excitement"
    answerer_style = "Speak calmly and clearly"
    
    audio_file = text_to_speech(dialogs, questioner_style, answerer_style)
"""

import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)


def text_to_speech(dialogs: List[str], questioner_tone_style: str, answerer_tone_style: str) -> str:
    """Convert dialogue text to speech audio files.
    
    Takes a list of dialogue lines and converts them to speech using OpenAI's
    text-to-speech API. Alternates between two different voices (sage and alloy)
    to represent different speakers, applies tone styling, and combines all
    audio segments into a single output file.
    
    Args:
        dialogs: List of dialogue lines to convert to speech
        questioner_tone_style: Style instruction for the questioner's voice
        answerer_tone_style: Style instruction for the answerer's voice
        
    Returns:
        Filename (without extension) of the combined audio file
        
    Raises:
        ValueError: If dialogs list is empty
        Exception: If audio generation or file operations fail
    """
    if not dialogs:
        raise ValueError("Dialogs list cannot be empty")
    
    # Voice mapping for alternating speakers
    speaker_voices = ["sage", "alloy"]
    voice_instructions = {
        "sage": questioner_tone_style,
        "alloy": answerer_tone_style,
    }
    
    # Generate unique identifier for this audio session
    unique_id = uuid.uuid4()
    generated_files = []
    
    logger.info(f"Starting text-to-speech conversion for {len(dialogs)} dialogue lines")
    
    try:
        # Generate individual audio files for each dialogue line
        for i, dialog_line in enumerate(dialogs):
            if not dialog_line.strip():
                logger.warning(f"Skipping empty dialogue line at index {i}")
                continue
                
            # Alternate between speakers
            current_voice = speaker_voices[i % 2]
            
            logger.debug(f"Generating audio for line {i+1}/{len(dialogs)} with voice '{current_voice}'")
            
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=current_voice,
                input=dialog_line,
                instructions=voice_instructions[current_voice],
            ) as response:
                filename = f"{unique_id}_{i}.mp3"
                file_path = f"speech_files/{filename}"
                response.stream_to_file(file_path)
                generated_files.append(filename)
                
        logger.info(f"Generated {len(generated_files)} individual audio files")
        
        # Combine all individual files into one
        combined_filename = str(unique_id)
        _combine_speech_files("speech_files", generated_files, combined_filename)
        
        logger.info(f"Successfully created combined audio file: {combined_filename}.mp3")
        return combined_filename
        
    except Exception as e:
        logger.error(f"Error during text-to-speech conversion: {e}")
        # Cleanup any partially generated files
        _cleanup_files("speech_files", generated_files)
        raise


def _combine_speech_files(folder_name: str, file_names: List[str], output_file_name: str) -> None:
    """Combine multiple audio files into a single output file with silence intervals.
    
    Reads individual audio files and combines them sequentially, adding silence
    between every other file (after answerer responses) to create natural pauses
    in the conversation.
    
    Args:
        folder_name: Directory containing the audio files
        file_names: List of audio filenames to combine
        output_file_name: Name for the combined output file (without extension)
        
    Raises:
        FileNotFoundError: If silence file or any input files are missing
        IOError: If file read/write operations fail
    """
    if not file_names:
        logger.warning("No files to combine")
        return
        
    silence_path = Path(folder_name) / "silent.mp3"
    output_path = Path(folder_name) / f"{output_file_name}.mp3"
    
    try:
        # Read silence audio data
        if not silence_path.exists():
            raise FileNotFoundError(f"Silence file not found: {silence_path}")
            
        with open(silence_path, "rb") as f:
            silence_binary_data = f.read()
            
        logger.info(f"Combining {len(file_names)} audio files into {output_path}")
        
        # Combine all audio files
        with open(output_path, "wb") as output_file:
            for i, filename in enumerate(file_names):
                input_path = Path(folder_name) / filename
                
                if not input_path.exists():
                    logger.error(f"Input file not found: {input_path}")
                    continue
                    
                # Write audio content
                with open(input_path, "rb") as input_file:
                    output_file.write(input_file.read())
                    
                # Add silence after every answerer response (odd indices)
                if i % 2 == 1 and i < len(file_names) - 1:  # Don't add silence after last file
                    output_file.write(silence_binary_data)
                    
        logger.info(f"Successfully combined audio files into {output_path}")
        # Remove individual files
        _cleanup_files(folder_name, file_names)
    except Exception as e:
        logger.error(f"Error combining audio files: {e}")
        raise


def _cleanup_files(folder_name: str, file_names: List[str]) -> None:
    """Clean up temporary audio files.
    
    Removes individual audio files that were generated during the text-to-speech
    process to avoid cluttering the speech_files directory.
    
    Args:
        folder_name: Directory containing the files to clean up
        file_names: List of filenames to remove
    """
    if not file_names:
        return
        
    logger.info(f"Cleaning up {len(file_names)} temporary audio files")
    
    for filename in file_names:
        try:
            file_path = Path(folder_name) / filename
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Removed temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {filename}: {e}")


if __name__ == "__main__":
    # filenames = ["6ac8fdda-7ac9-4c1c-bfa8-dfe9b4b4a0f4_0.mp3",
    # "6ac8fdda-7ac9-4c1c-bfa8-dfe9b4b4a0f4_1.mp3",
    # "6ac8fdda-7ac9-4c1c-bfa8-dfe9b4b4a0f4_2.mp3",
    # "6ac8fdda-7ac9-4c1c-bfa8-dfe9b4b4a0f4_3.mp3",
    # "6ac8fdda-7ac9-4c1c-bfa8-dfe9b4b4a0f4_4.mp3",
    # "6ac8fdda-7ac9-4c1c-bfa8-dfe9b4b4a0f4_5.mp3",
    # "6ac8fdda-7ac9-4c1c-bfa8-dfe9b4b4a0f4_6.mp3",
    # "6ac8fdda-7ac9-4c1c-bfa8-dfe9b4b4a0f4_7.mp3",
    # "6ac8fdda-7ac9-4c1c-bfa8-dfe9b4b4a0f4_8.mp3",
    # "6ac8fdda-7ac9-4c1c-bfa8-dfe9b4b4a0f4_9.mp3",
    # "6ac8fdda-7ac9-4c1c-bfa8-dfe9b4b4a0f4_10.mp3",
    # "6ac8fdda-7ac9-4c1c-bfa8-dfe9b4b4a0f4_11.mp3",
    # ]
    # _cleanup_files("speech_files", filenames)
    pass
