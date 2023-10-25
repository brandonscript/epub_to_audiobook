import os
import re

import io
import tempfile
import wave
from scipy.io.wavfile import write
import argparse
import html
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import requests
from typing import List, Tuple
from datetime import datetime, timedelta
from mutagen.mp3 import MP3
from mutagen.id3 import TIT2, TPE1, TALB, TRCK
import logging
from time import sleep
import torch
from TTS.api import TTS
import scipy.io.wavfile as wavfile
from pydub import AudioSegment


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Added max_retries constant
MAX_RETRIES = 12

MAGIC_BREAK_STRING = " @BRK#"  # leading blank is for text split

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/en/jenny/jenny").to(device)


def sanitize_title(title: str) -> str:
    # replace MAGIC_BREAK_STRING with a blank space
    # strip incase leading bank is missing
    title = title.replace(MAGIC_BREAK_STRING.strip(), " ")
    sanitized_title = re.sub(r"[^\w\s]", "", title, flags=re.UNICODE)
    sanitized_title = re.sub(r"\s+", "_", sanitized_title.strip())
    return sanitized_title


def extract_chapters(epub_book: epub.EpubBook, newline_mode: str, remove_endnotes: bool) -> List[Tuple[str, str]]:
    chapters = []
    for item in epub_book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content()
            soup = BeautifulSoup(content, features='lxml-xml')
            title = soup.title.string if soup.title else ''
            raw = soup.get_text(strip=False)
            logger.debug(f"Raw text: <{raw[:]}>")

            # Replace excessive whitespaces and newline characters based on the mode
            if newline_mode == 'single':
                cleaned_text = re.sub(
                    r'[\n]+', MAGIC_BREAK_STRING, raw.strip())
            elif newline_mode == 'double':
                cleaned_text = re.sub(
                    r'[\n]{2,}', MAGIC_BREAK_STRING, raw.strip())
            else:
                raise ValueError(f"Invalid newline mode: {newline_mode}")

            logger.debug(f"Cleaned text step 1: <{cleaned_text[:]}>")
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            logger.info(f"Cleaned text step 2: <{cleaned_text[:100]}>")
            
            #Removes endnote numbers
            if remove_endnotes == True:
                cleaned_text = re.sub(r'(?<=[a-zA-Z.,!?;”")])\d+', '', cleaned_text)
                logger.info(f"Cleaned text step 4: <{cleaned_text[:100]}>")

            # fill in the title if it's missing
            if not title:
                title = cleaned_text[:60]
            logger.debug(f"Raw title: <{title}>")
            title = sanitize_title(title)
            logger.info(f"Sanitized title: <{title}>")

            chapters.append((title, cleaned_text))
            soup.decompose()

    return chapters


def is_special_char(char: str) -> bool:
    # Check if the character is a English letter, number or punctuation or a punctuation in Chinese, never split these characters.
    ord_char = ord(char)
    result = (ord_char >= 33 and ord_char <= 126) or (
        char in "。，、？！：；“”‘’（）《》【】…—～·「」『』〈〉〖〗〔〕") or (
            char in "∶")  # special unicode punctuation
    logger.debug(
        f"is_special_char> char={char}, ord={ord_char}, result={result}")
    return result


def split_text(text: str, max_chars: int) -> List[str]:
    chunks = []
    current_chunk = ""
    words = text.split()

    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + word
        else:
            chunks.append(current_chunk)
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    logger.info(f"Split text into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks, 1):
        first_100 = chunk[:100]
        last_100 = chunk[-100:] if len(chunk) > 100 else ""
        logger.info(
            f"Chunk {i}: Length={len(chunk)}, Start={first_100}..., End={last_100}")

    return chunks


def convert_wav_to_mp3(wav_path, mp3_path):
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")


def tts_to_bytes_io(text):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file_path = temp_file.name

        # Call tts_to_file to generate the audio
        tts.tts_to_file(text=text, file_path=temp_file_path)

        # get mp3 version of the file at temp_file_path with the same name, but with .mp3 extension
        mp3_path = temp_file_path.replace(".wav", ".mp3")
        convert_wav_to_mp3(temp_file_path, mp3_path)

        # Read the temporary file
        with open(mp3_path, 'rb') as f:
            audio_data = f.read()

        # echo temp file paths for debugging
        # print(f"temp_file_path: {temp_file_path}")
        # print(f"mp3_path: {mp3_path}")

        # # sleep for 10 seconds to allow for debugging
        # sleep(10)

        # Clean up the temporary file
        os.remove(temp_file_path)
        os.remove(mp3_path)

    return audio_data


def text_to_speech(session: requests.Session, text: str, output_file: str, tts_model: str, break_duration: int, title: str, author: str, book_title: str, idx: int, output_format: str, sample: bool = False) -> None:
    # Adjust this value based on your testing
    max_chars = 3000

    text_chunks = split_text(text, max_chars)

    audio_segments = []

    for i, chunk in enumerate(text_chunks, 1):
        logger.debug(
            f"Processing chunk {i} of {len(text_chunks)}, length={len(chunk)}, text=[{chunk}]")
        escaped_text = html.escape(chunk)
        logger.debug(f"Escaped text: [{escaped_text}]")
        # replace MAGIC_BREAK_STRING with a break tag for section/paragraph break
        escaped_text = escaped_text.replace(
            MAGIC_BREAK_STRING.strip(), f" <break time='{break_duration}ms' /> ")  # strip in case leading bank is missing
        logger.info(
            f"Processing chapter-{idx} <{title}>, chunk {i} of {len(text_chunks)}")

        audio_segments.append(io.BytesIO(tts_to_bytes_io(escaped_text)))

        if sample:
            break

    with open(output_file, "wb") as outfile:
        for segment in audio_segments:
            segment.seek(0)
            outfile.write(segment.read())

    # Add ID3 tags to the generated MP3 file
    audio = MP3(output_file)
    audio["TIT2"] = TIT2(encoding=3, text=title)
    audio["TPE1"] = TPE1(encoding=3, text=author)
    audio["TALB"] = TALB(encoding=3, text=book_title)
    audio["TRCK"] = TRCK(encoding=3, text=str(idx))
    audio.save()


def epub_to_audiobook(input_file: str, output_folder: str, tts_model: str, preview: bool, newline_mode: str, break_duration: int, chapter_start: int, chapter_end: int, output_format: str, remove_endnotes: bool, output_text: bool, sample: bool = False) -> None:
    book = epub.read_epub(input_file)
    chapters = extract_chapters(book, newline_mode, remove_endnotes)

    os.makedirs(output_folder, exist_ok=True)

    # Get the book title and author from metadata or use fallback values
    book_title = "Untitled"
    author = "Unknown"
    if book.get_metadata('DC', 'title'):
        book_title = book.get_metadata('DC', 'title')[0][0]
    if book.get_metadata('DC', 'creator'):
        author = book.get_metadata('DC', 'creator')[0][0]

    # Filter out empty or very short chapters
    chapters = [(title, text) for title, text in chapters if text.strip()]

    logger.info(f"Chapters count: {len(chapters)}.")

    # Check chapter start and end args
    if chapter_start < 1 or chapter_start > len(chapters):
        raise ValueError(
            f"Chapter start index {chapter_start} is out of range. Check your input.")
    if chapter_end < -1 or chapter_end > len(chapters):
        raise ValueError(
            f"Chapter end index {chapter_end} is out of range. Check your input.")
    if chapter_end == -1:
        chapter_end = len(chapters)
    if chapter_start > chapter_end:
        raise ValueError(
            f"Chapter start index {chapter_start} is larger than chapter end index {chapter_end}. Check your input.")

    logger.info(f"Converting chapters {chapter_start} to {chapter_end}.")

    with requests.Session() as session:
        for idx, (title, text) in enumerate(chapters, start=1):
            if idx < chapter_start:
                continue
            if idx > chapter_end:
                break
            logger.info(f"Converting chapter {idx}/{len(chapters)}: {title}")
            if preview:
                continue
            
            if output_text:
                text_file = os.path.join(output_folder, f"{idx:04d}_{title}.txt")
                with open(text_file, "w") as file:
                    file.write(text)
            
            output_file = os.path.join(output_folder, f"{idx:04d}_{title}.mp3")
            text_to_speech(session, text, output_file, tts_model,
                           break_duration, title, author, book_title, idx, output_format, sample)

            if sample:
                break


def main():
    global tts
    parser = argparse.ArgumentParser(description="Convert EPUB to audiobook")
    parser.add_argument("input_file", help="Path to the EPUB file",
                        default="examples/RobinsonCrusoe.epub")
    parser.add_argument("output_folder", help="Path to the output folder",
                        default="./examples/RobinsonCrusoe")
    parser.add_argument("model", help="Path to the TTS model",
                        default="tts_models/en/jenny/jenny")
    parser.add_argument("--voice_name", default="en-US-GuyNeural",
                        help="Voice name for the text-to-speech service (default: en-US-GuyNeural). You can use zh-CN-YunyeNeural for Chinese ebooks.")
    parser.add_argument("--language", default="en-US",
                        help="Language for the text-to-speech service (default: en-US)")
    parser.add_argument("--log", default="INFO",
                        help="Log level (default: INFO), can be DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument("--preview", action="store_true",
                        help="Enable preview mode. In preview mode, the script will not convert the text to speech. Instead, it will print the chapter index and titles.")
    parser.add_argument("--sample", action="store_true",
                        help="Enable sample mode. In sample mode, the script will only convert one set of <= 3,000 characters. This is useful for testing.")
    parser.add_argument('--newline_mode', choices=['single', 'double'], default='double',
                        help="Choose the mode of detecting new paragraphs: 'single' or 'double'. 'single' means a single newline character, while 'double' means two consecutive newline characters. (default: double, works for most ebooks but will detect less paragraphs for some ebooks)")
    parser.add_argument("--break_duration", default="1250",
                        help="Break duration in milliseconds for the different paragraphs or sections (default: 1250). Valid values range from 0 to 5000 milliseconds.")
    parser.add_argument("--chapter_start", default=1, type=int,
                        help="Chapter start index (default: 1, starting from 1)")
    parser.add_argument("--chapter_end", default=-1, type=int,
                        help="Chapter end index (default: -1, meaning to the last chapter)")
    parser.add_argument("--output_format", default="audio-24khz-48kbitrate-mono-mp3", help="Output format for the text-to-speech service (default: audio-24khz-48kbitrate-mono-mp3). Support formats: audio-16khz-32kbitrate-mono-mp3 audio-16khz-64kbitrate-mono-mp3 audio-16khz-128kbitrate-mono-mp3 audio-24khz-48kbitrate-mono-mp3 audio-24khz-96kbitrate-mono-mp3 audio-24khz-160kbitrate-mono-mp3 audio-48khz-96kbitrate-mono-mp3 audio-48khz-192kbitrate-mono-mp3. See https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech?tabs=streaming#audio-outputs. Only mp3 is supported for now. Different formats will result in different audio quality and file size.")
    parser.add_argument("--output_text", action="store_true", help="Enable Output Text. This will export a plain text file for each chapter specified and write the files to the output foler specified.")
    parser.add_argument("--remove_endnotes", action="store_true", help="This will remove endnote numbers from the end or middle of sentences. This is useful for academic books.")

    args = parser.parse_args()

    logger.setLevel(args.log)

    tts = TTS(args.model).to(device)

    epub_to_audiobook(args.input_file, args.output_folder,
                      args.model, args.preview, args.newline_mode, args.break_duration, args.chapter_start, args.chapter_end, args.output_format, args.remove_endnotes, args.output_text, args.sample)
    logger.info("Done! 👍")
    logger.info(f"args = {args}")


if __name__ == "__main__":
    main()
