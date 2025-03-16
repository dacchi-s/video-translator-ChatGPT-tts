import os
import logging
from typing import List, Dict, Any
import torch
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, ColorClip, AudioFileClip
import whisper
import srt
from datetime import timedelta
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import tiktoken
import textwrap
import argparse
from openai import OpenAI
from pydub import AudioSegment
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Paths
    FONT_PATH = os.getenv('FONT_PATH', "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    JAPANESE_FONT_PATH = os.getenv('JAPANESE_FONT_PATH', "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    TEMP_AUDIO_FILE = os.getenv('TEMP_AUDIO_FILE', "temp_audio.wav")

    # Video processing
    DEFAULT_SUBTITLE_HEIGHT = int(os.getenv('DEFAULT_SUBTITLE_HEIGHT', 200))
    DEFAULT_FONT_SIZE = int(os.getenv('DEFAULT_FONT_SIZE', 32))
    MAX_SUBTITLE_LINES = int(os.getenv('MAX_SUBTITLE_LINES', 3))

    # Video encoding
    VIDEO_CODEC = os.getenv('VIDEO_CODEC', 'libx264')
    AUDIO_CODEC = os.getenv('AUDIO_CODEC', 'aac')
    VIDEO_PRESET = os.getenv('VIDEO_PRESET', 'medium')
    CRF = os.getenv('CRF', '23')
    PIXEL_FORMAT = os.getenv('PIXEL_FORMAT', 'yuv420p')

    # Tiktoken related settings
    TIKTOKEN_MODEL = "cl100k_base"
    MAX_TOKENS_PER_CHUNK = 4000

    # OpenAI API settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    TTS_MODEL = os.getenv('TTS_MODEL', "tts-1-hd")
    TTS_VOICE = os.getenv('TTS_VOICE', "echo")

    # GPT-4 translation settings
    DEFAULT_GPT_MODEL = "gpt-4o"
    GPT_MAX_TOKENS = 4000

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSProcessor:
    def __init__(self, api_key: str, tts_model: str = Config.TTS_MODEL, tts_voice: str = Config.TTS_VOICE):
        self.client = OpenAI(api_key=api_key)
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        logger.info(f"Initialized TTS Processor with model: {tts_model}, voice: {tts_voice}")

    def text_to_speech(self, text: str, output_file: str):
        try:
            # Using the with_streaming_response approach as recommended in the warning
            with self.client.audio.speech.with_streaming_response.create(
                model=self.tts_model,
                voice=self.tts_voice,
                input=text
            ) as response:
                # Open the output file in binary write mode
                with open(output_file, 'wb') as f:
                    # Iterate through the response chunks and write them to the file
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                        
            logger.info(f"TTS audio saved to {output_file}")
        except Exception as e:
            logger.error(f"Error in TTS conversion: {e}")
            raise

class SubtitleProcessor:
    def __init__(self, video_path: str, srt_path: str):
        self.video_path = video_path
        self.srt_path = srt_path
        self.temp_files = []

    def cleanup_temp_files(self):
        logger.info("Cleaning up temporary files...")
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing {file_path}: {e}")
        self.temp_files.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_temp_files()

class SRTGenerator(SubtitleProcessor):
    def __init__(self, video_path: str, output_srt: str, model_name: str, translate: bool = False, language: str = "Japanese"):
        super().__init__(video_path, output_srt)
        self.model_name = model_name
        self.translate = translate
        self.language = language
        self.tokenizer = tiktoken.get_encoding(Config.TIKTOKEN_MODEL)

    def run(self):
        try:
            self.extract_audio()
            transcription = self.transcribe_audio()
            chunks = self.split_into_chunks(transcription)
            results = self.process_chunks(chunks)
            self.create_srt(results)
            logger.info(f"SRT file has been generated: {self.srt_path}")
        finally:
            self.cleanup_temp_files()

    def extract_audio(self):
        logger.info("Extracting audio from video...")
        video = VideoFileClip(self.video_path)
        video.audio.write_audiofile(Config.TEMP_AUDIO_FILE)
        self.temp_files.append(Config.TEMP_AUDIO_FILE)

    def transcribe_audio(self) -> Dict[str, Any]:
        logger.info("Transcribing audio with Whisper...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        logger.info(f"Loading Whisper model: {self.model_name}")
        model = whisper.load_model(self.model_name).to(device)

        task = "translate" if self.translate else "transcribe"
        logger.info(f"Performing task: {task}")
        result = model.transcribe(Config.TEMP_AUDIO_FILE, task=task, language=self.language)
        return result

    def split_into_chunks(self, transcription: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.info("Splitting transcription into chunks...")
        chunks = []
        current_chunk = {"text": "", "segments": []}
        current_tokens = 0

        for segment in transcription['segments']:
            segment_tokens = self.tokenizer.encode(segment['text'])
            if current_tokens + len(segment_tokens) > Config.MAX_TOKENS_PER_CHUNK:
                chunks.append(current_chunk)
                current_chunk = {"text": "", "segments": []}
                current_tokens = 0
            
            current_chunk['text'] += segment['text'] + " "
            current_chunk['segments'].append(segment)
            current_tokens += len(segment_tokens)

        if current_chunk['segments']:
            chunks.append(current_chunk)

        return chunks

    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info("Processing chunks...")
        results = []
        for chunk in tqdm(chunks, desc="Processing chunks"):
            results.extend(chunk['segments'])
        return results

    def create_srt(self, results: List[Dict[str, Any]]):
        logger.info("Creating SRT file...")
        subs = []
        for i, segment in enumerate(results, start=1):
            start = timedelta(seconds=segment['start'])
            end = timedelta(seconds=segment['end'])
            text = segment['text']
            sub = srt.Subtitle(index=i, start=start, end=end, content=text)
            subs.append(sub)
        
        with open(self.srt_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(subs))

class SubtitleAdder(SubtitleProcessor):
    def __init__(self, video_path: str, output_video: str, input_srt: str, 
                 subtitle_height: int = Config.DEFAULT_SUBTITLE_HEIGHT, 
                 add_tts: bool = False,
                 tts_model: str = Config.TTS_MODEL,
                 tts_voice: str = Config.TTS_VOICE,
                 audio_mode: str = "replace"):
        super().__init__(video_path, input_srt)
        self.output_video = output_video
        self.subtitle_height = subtitle_height
        self.add_tts = add_tts
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.audio_mode = audio_mode
        self.tts_processor = TTSProcessor(Config.OPENAI_API_KEY, tts_model, tts_voice) if add_tts else None

    def run(self):
        try:
            subs = self.load_srt(self.srt_path)
            if self.add_tts:
                self.generate_tts_audio(subs)
            self.add_subtitles_to_video(subs)
            logger.info(f"Video processing complete: {self.output_video}")
        finally:
            self.cleanup_temp_files()

    @staticmethod
    def load_srt(srt_path: str) -> List[srt.Subtitle]:
        logger.info(f"Loading SRT file: {srt_path}")
        with open(srt_path, 'r', encoding='utf-8') as f:
            return list(srt.parse(f.read()))

    def generate_tts_audio(self, subs: List[srt.Subtitle]):
        logger.info("Generating TTS audio...")
        combined_audio = AudioSegment.silent(duration=0)

        for sub in tqdm(subs, desc="Generating TTS"):
            tts_file = f"temp_tts_{sub.index}.mp3"
            self.tts_processor.text_to_speech(sub.content, tts_file)
            self.temp_files.append(tts_file)

            segment = AudioSegment.from_mp3(tts_file)
            silence_duration = int(sub.start.total_seconds() * 1000) - len(combined_audio)
            if silence_duration > 0:
                combined_audio += AudioSegment.silent(duration=silence_duration)
            combined_audio += segment

        combined_audio.export("combined_tts.mp3", format="mp3")
        self.temp_files.append("combined_tts.mp3")

    def add_subtitles_to_video(self, subs: List[srt.Subtitle], audio_mode="replace"):
        logger.info(f"Adding subtitles to video with subtitle space height of {self.subtitle_height} pixels...")
        video = VideoFileClip(self.video_path)
        
        original_width, original_height = video.w, video.h
        new_height = original_height + self.subtitle_height
        
        background = ColorClip(size=(original_width, new_height), color=(0,0,0), duration=video.duration)
        video_clip = video.set_position((0, 0))
        
        # Create progress bar for subtitle generation
        logger.info("Generating subtitle clips...")
        subtitle_clips = []
        
        for sub in tqdm(subs, desc="Creating subtitle clips"):
            clip = self.create_subtitle_clip(sub.content, original_width) \
                .set_start(sub.start.total_seconds()) \
                .set_end(sub.end.total_seconds()) \
                .set_position((0, original_height))
            subtitle_clips.append(clip)
        
        logger.info("Compositing video with subtitles...")
        final_video = CompositeVideoClip([background, video_clip] + subtitle_clips, size=(original_width, new_height))
        final_video = final_video.set_duration(video.duration)
        
        if self.add_tts and os.path.exists("combined_tts.mp3"):
            logger.info("Adding TTS audio to the video...")
            try:
                tts_audio = AudioFileClip("combined_tts.mp3")
                
                if self.audio_mode == "mix" and video.audio:
                    logger.info("Mixing TTS audio with original audio...")
                    tts_audio = tts_audio.volumex(0.7)
                    final_audio = CompositeAudioClip([video.audio, tts_audio])
                else:
                    logger.info("Replacing original audio with TTS audio...")
                    final_audio = tts_audio
                    
                final_video = final_video.set_audio(final_audio)
                logger.info("TTS audio added successfully")
            except Exception as e:
                logger.error(f"Error adding TTS audio: {e}")
        
        logger.info("Rendering final video with subtitles (this may take a while)...")
        final_video.write_videofile(
            self.output_video, 
            codec=Config.VIDEO_CODEC, 
            audio_codec=Config.AUDIO_CODEC,
            preset=Config.VIDEO_PRESET,
            ffmpeg_params=['-crf', Config.CRF, '-pix_fmt', Config.PIXEL_FORMAT],
            verbose=True,
            logger="bar"
        )
        
        logger.info("Video rendering complete!")

    @staticmethod
    def create_subtitle_clip(txt: str, video_width: int, font_size: int = Config.DEFAULT_FONT_SIZE, max_lines: int = Config.MAX_SUBTITLE_LINES) -> ImageClip:
        if any(ord(char) > 127 for char in txt):
            font_path = Config.JAPANESE_FONT_PATH
        else:
            font_path = Config.FONT_PATH

        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            logger.warning(f"Failed to load font from {font_path}. Falling back to default font.")
            font = ImageFont.load_default()
        
        max_char_count = int(video_width / (font_size * 0.6))
        wrapped_text = textwrap.fill(txt, width=max_char_count)
        lines = wrapped_text.split('\n')[:max_lines]
        
        dummy_img = Image.new('RGB', (video_width, font_size * len(lines)))
        dummy_draw = ImageDraw.Draw(dummy_img)
        max_line_width = max(dummy_draw.textbbox((0, 0), line, font=font)[2] for line in lines)
        total_height = sum(dummy_draw.textbbox((0, 0), line, font=font)[3] for line in lines)
        
        img_width, img_height = video_width, total_height + 20
        img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        y_text = 10
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            x_text = (img_width - bbox[2]) // 2
            
            for adj in range(-2, 3):
                for adj2 in range(-2, 3):
                    draw.text((x_text+adj, y_text+adj2), line, font=font, fill=(0, 0, 0, 255))
            
            draw.text((x_text, y_text), line, font=font, fill=(255, 255, 255, 255))
            y_text += bbox[3]
        
        return ImageClip(np.array(img))

class SRTTranslator:
    def __init__(self, api_key: str, model: str = Config.DEFAULT_GPT_MODEL, temperature: float = 0.3):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def translate_srt(self, input_srt: str, output_srt: str, source_lang: str, target_lang: str):
        with open(input_srt, 'r', encoding='utf-8') as f:
            subtitle_generator = srt.parse(f.read())
            subtitles = list(subtitle_generator)

        translated_subtitles = []
        for subtitle in tqdm(subtitles, desc="Translating subtitles"):
            translated_content = self.translate_text(subtitle.content, source_lang, target_lang)
            translated_subtitle = srt.Subtitle(
                index=subtitle.index,
                start=subtitle.start,
                end=subtitle.end,
                content=translated_content
            )
            translated_subtitles.append(translated_subtitle)

        with open(output_srt, 'w', encoding='utf-8') as f:
            f.write(srt.compose(translated_subtitles))

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. Maintain the original meaning and nuance as much as possible. Do not modify any formatting or line breaks."},
                {"role": "user", "content": text}
            ],
            temperature=self.temperature,
            max_tokens=Config.GPT_MAX_TOKENS
        )
        return response.choices[0].message.content.strip()

def main():
    parser = argparse.ArgumentParser(description="Subtitle Generator and Adder with TTS and GPT Translation", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Generate subparser
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--input", required=True, help="Input video file path")
    generate_parser.add_argument("--output_srt", required=True, help="Output SRT file path")
    generate_parser.add_argument("--model", default="large-v3", help="Whisper model name (default: large-v3)")
    generate_parser.add_argument("--translate", action="store_true", help="Translate the audio to English using Whisper")
    generate_parser.add_argument("--language", default="Japanese", help="Specify the language of the audio (default: Japanese)")

    # Add subparser
    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("--input", required=True, help="Input video file path")
    add_parser.add_argument("--output_video", required=True, help="Output video file path")
    add_parser.add_argument("--input_srt", required=True, help="Input SRT file path")
    add_parser.add_argument("--mode", choices=["subtitles", "tts", "both"], default="both", 
                            help="Mode of operation: 'subtitles' for English subtitles only, 'tts' for English TTS audio only, or 'both' for both (default)")
    add_parser.add_argument("--tts_model", choices=["tts-1", "tts-1-hd"], default=Config.TTS_MODEL, 
                           help="TTS model to use: 'tts-1' (faster) or 'tts-1-hd' (higher quality) (default: tts-1-hd)")
    add_parser.add_argument("--tts_voice", choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "ash", "sage"], 
                           default=Config.TTS_VOICE, help="TTS voice to use (default: echo)")
    add_parser.add_argument("--audio_mode", choices=["replace", "mix"], default="replace",
                       help="How to handle TTS audio: 'replace' original audio (default) or 'mix' with it")

    # Translate subparser
    translate_parser = subparsers.add_parser("translate")
    translate_parser.add_argument("--input_srt", required=True, help="Input SRT file path")
    translate_parser.add_argument("--output_srt", required=True, help="Output translated SRT file path")
    translate_parser.add_argument("--source_lang", default="Japanese", help="Source language (default: Japanese)")
    translate_parser.add_argument("--target_lang", default="English", help="Target language (default: English)")
    translate_parser.add_argument("--temperature", type=float, default=0.3, help="GPT temperature (default: 0.3)")
    translate_parser.add_argument("--gpt_model", default=Config.DEFAULT_GPT_MODEL, help=f"GPT model to use (default: {Config.DEFAULT_GPT_MODEL})")

    args = parser.parse_args()

    if args.action == "generate":
        with SRTGenerator(args.input, args.output_srt, args.model, args.translate, args.language) as generator:
            generator.run()
    elif args.action == "add":
        add_subtitles = args.mode in ["subtitles", "both"]
        add_tts = args.mode in ["tts", "both"]
        
        with SubtitleAdder(
            args.input, 
            args.output_video, 
            args.input_srt, 
            add_tts=add_tts,
            tts_model=args.tts_model,
            tts_voice=args.tts_voice,
            audio_mode=args.audio_mode
        ) as adder:
            if add_subtitles:
                adder.run()
            else:
                subs = adder.load_srt(args.input_srt)
                adder.generate_tts_audio(subs)
                video = VideoFileClip(args.input)
                tts_audio = AudioFileClip("combined_tts.mp3")
                
                if args.audio_mode == "mix" and video.audio:
                    tts_audio = tts_audio.volumex(0.7)
                    final_audio = CompositeAudioClip([video.audio, tts_audio])
                else:
                    final_audio = tts_audio
                    
                final_video = video.set_audio(final_audio)
                final_video.write_videofile(
                    args.output_video, 
                    codec=Config.VIDEO_CODEC, 
                    audio_codec=Config.AUDIO_CODEC
                )
    elif args.action == "translate":
        translator = SRTTranslator(api_key=Config.OPENAI_API_KEY, model=args.gpt_model, temperature=args.temperature)
        translator.translate_srt(args.input_srt, args.output_srt, args.source_lang, args.target_lang)


if __name__ == "__main__":
    main()
