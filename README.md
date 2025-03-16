# Video Translation & Text-to-Speech Tool

A powerful AI tool that translates Japanese videos into English (or other languages) and adds natural-sounding voice narration. This tool automatically generates subtitles from videos, translates them, and converts them to speech using Text-to-Speech (TTS) technology.

## Main Features

- **Automatic Transcription**: Generate accurate subtitle files (SRT) from videos using OpenAI Whisper
- **High-Quality Translation**: Translate subtitles to English or other languages using OpenAI GPT models
- **Natural Voice Synthesis**: Convert translated text to natural-sounding speech using OpenAI TTS
- **Seamless Integration**: Add both subtitles and TTS audio to videos with customizable formatting
- **GPU Acceleration**: Use CUDA for faster speech recognition processing

## System Requirements

- NVIDIA graphics card with CUDA support (recommended for faster processing)
- Docker and NVIDIA Container Toolkit
- OpenAI API key (needed for translation and TTS features)
- Python 3.9 or newer

## Docker Setup

### 1. Install Docker Engine

```bash
# Remove old Docker installations if necessary
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do 
  sudo apt remove $pkg
done

# Install Docker
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Test installation
sudo docker run hello-world
```

### 2. Configure Docker for Non-Root Users

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
wsl --shutdown  # Only if using WSL
```

### 3. NVIDIA Docker Setup (for GPU support)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Creating and Setting Up the Docker Container

### 1. Pull CUDA Docker Image

```bash
docker pull nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
docker run -it --gpus all nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```

### 2. Install Dependencies

```bash
apt update && apt full-upgrade -y
apt install git wget nano ffmpeg -y
```

### 3. Install Miniconda

```bash
cd ~
mkdir tmp
cd tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts: yes, enter, yes

# Clean up temp folder
cd ..
rm -rf tmp

# Exit and restart container
exit
docker container ls -a
docker start <container id>
docker exec -it <container id> /bin/bash
```

### 4. Create Conda Environment

```bash
mkdir text-to-speech
cd text-to-speech
nano text-to-speech.yml
```

Add the following content to subtitle-generator.yml:

```yaml
name: text-to-speech
channels:
  - conda-forge
  - pytorch
  - nvidia
  - defaults
dependencies:
  - python=3.9
  - pip
  - cudatoolkit=11.8
  - tiktoken
  - pillow
  - tqdm
  - srt
  - moviepy
  - python-dotenv
  - pip:
    - pydub==0.25.1
    - openai
    - openai-whisper
    - torch
    - torchvision
    - torchaudio
```

Create the environment:

```bash
conda env create -f text-to-speech.yml
conda activate text-to-speech
```

## Script Setup

1. Copy the Python script to your container:

```bash
nano text-to-speech.py
# Paste the script content and save
```

2. Create a `.env` file with your OpenAI API key and other settings:

```bash
nano .env
# Add the following content:
OPENAI_API_KEY=your_openai_api_key
```

## TTS Models and Voices

### TTS Models
The script supports two TTS model options from OpenAI:

- **`tts-1`** (default): Good for realtime applications with lower latency. May have some static in certain situations.
- **`tts-1-hd`**: Higher quality audio with better clarity, but slightly slower processing time.

### Available TTS Voices

This script supports these TTS voices from OpenAI:
- `alloy` (default) - Balanced, neutral voice
- `echo` - Soft, warm voice
- `fable` - Authoritative, expressive voice
- `onyx` - Deep, powerful voice
- `nova` - Cheerful, upbeat voice
- `shimmer` - Clear, professional voice
- `coral` - Mature, melodious voice
- `ash` - Balanced, thoughtful voice
- `sage` - Calming, measured voice

All voices are optimized for English. You can experiment with different voices to find the best match for your desired tone.

### Audio Handling Options

When adding TTS to your video, you can control how the generated voice interacts with the original audio:

- **`replace`** (default): Completely replaces the original audio with the TTS audio
- **`mix`**: Mixes the TTS audio with the original audio (TTS at 70% volume)

## Usage Examples

### 1. Create English Subtitles and Voice from a Japanese Video

```bash
# Step 1: Generate Japanese subtitles from the video
python text-to-speech.py generate --input japanese_video.mp4 --output_srt japanese_subs.srt --model large-v3

# Step 2: Translate Japanese subtitles to English
python text-to-speech.py translate --input_srt japanese_subs.srt --output_srt english_subs.srt

# Step 3: Add English subtitles and voice to the video
python text-to-speech.py add --input japanese_video.mp4 --output_video english_version.mp4 --input_srt english_subs.srt --mode both
```

### 2. Add Subtitles Only (No Voice)

```bash
python text-to-speech.py add --input japanese_video.mp4 --output_video video_with_english_subs.mp4 --input_srt english_subs.srt --mode subtitles
```

### 3. Add Voice Only (No Subtitles)

```bash
python text-to-speech.py add --input japanese_video.mp4 --output_video video_with_english_voice.mp4 --input_srt english_subs.srt --mode tts
```

### 4. Add Voice and Mix with Original Audio

```bash
python text-to-speech.py add --input japanese_video.mp4 --output_video video_with_mixed_audio.mp4 --input_srt english_subs.srt --mode tts --audio_mode mix
```

## Command Reference

```
Usage: text-to-speech.py [-h] {generate,add,translate} ...

Subtitle Generation, Translation, and Text-to-Speech Tool
```

### Generate Command

```
python text-to-speech.py generate [-h] --input INPUT --output_srt OUTPUT_SRT [--model MODEL] [--translate] [--language LANGUAGE]

Options:
  --input INPUT         Input video file path
  --output_srt OUTPUT_SRT
                        Output SRT file path
  --model MODEL         Whisper model name (default: large-v3)
  --translate           Translate audio to English using Whisper
  --language LANGUAGE   Specify the language of the audio (default: Japanese)
```

### Add Command

```
python text-to-speech.py add [-h] --input INPUT --output_video OUTPUT_VIDEO --input_srt INPUT_SRT 
                               [--mode {subtitles,tts,both}] [--tts_model {tts-1,tts-1-hd}] 
                               [--tts_voice {alloy,echo,fable,onyx,nova,shimmer,coral,ash,sage}]
                               [--audio_mode {replace,mix}]

Options:
  --input INPUT         Input video file path
  --output_video OUTPUT_VIDEO
                        Output video file path
  --input_srt INPUT_SRT
                        Input SRT file path
  --mode {subtitles,tts,both}
                        Operation mode: 'subtitles' for subtitles only,
                        'tts' for TTS audio only,
                        'both' for both (default)
  --tts_model {tts-1,tts-1-hd}
                        TTS model to use: 'tts-1' (faster) or 'tts-1-hd' (higher quality)
                        (default: tts-1-hd)
  --tts_voice {alloy,echo,fable,onyx,nova,shimmer,coral,ash,sage}
                        TTS voice to use (default: echo)
  --audio_mode {replace,mix}
                        How to handle TTS audio: 'replace' original audio (default) or 'mix' with it
```

### Translate Command

```
python text-to-speech.py translate [-h] --input_srt INPUT_SRT --output_srt OUTPUT_SRT [--source_lang SOURCE_LANG] [--target_lang TARGET_LANG] [--temperature TEMPERATURE] [--gpt_model GPT_MODEL]

Options:
  --input_srt INPUT_SRT
                        Input SRT file path
  --output_srt OUTPUT_SRT
                        Output translated SRT file path
  --source_lang SOURCE_LANG
                        Source language (default: Japanese)
  --target_lang TARGET_LANG
                        Target language (default: English)
  --temperature TEMPERATURE
                        GPT temperature setting (default: 0.3)
  --gpt_model GPT_MODEL
                        GPT model to use (default: gpt-4o)
```

## Technical Details

### Components

1. **SRTGenerator**: Uses Whisper AI to create subtitle files (SRT) from videos
2. **SRTTranslator**: Uses OpenAI's GPT models to translate subtitles
3. **TTSProcessor**: Uses OpenAI's TTS API to convert text to speech
4. **SubtitleAdder**: Adds subtitles and TTS audio to videos

### Key Features

- **High-Quality Speech Recognition**: CUDA-accelerated Japanese speech recognition using Whisper models
- **Professional Translation**: Natural translation using the latest models like GPT-4o
- **Natural Voice Synthesis**: High-quality voice generation using OpenAI's TTS
- **Flexible Output Options**: Generate subtitles only, voice only, or both
- **Progress Visualization**: Progress bars for long-running processes
- **Audio Control**: Choose to replace original audio or mix it with the TTS audio

## File Transfer (Between Docker Container and PC)

```bash
# Copy from local machine to container
docker cp "/local_path/video.mp4" container_name:root/subtitle_generator/

# Copy from container to local machine
docker cp container_name:root/subtitle_generator/output_video.mp4 "/local_path/"
```

## License

[MIT License](LICENSE)

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper): Speech recognition
- [OpenAI API](https://openai.com/api/): Translation and TTS
- [MoviePy](https://zulko.github.io/moviepy/): Video processing
- [PyDub](https://github.com/jiaaro/pydub): Audio processing
