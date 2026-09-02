#!/usr/bin/env python3
"""
================================================================================
 YouTube Shorts Automation Pipeline
================================================================================
End-to-end pipeline that:
  1. Generates a "Did you know?" fun fact + title/description/tags/thumbnail
     prompt via a free LLM (g4f, falling back to Groq's free API).
  2. Turns the fact into a voiceover with edge-tts.
  3. Transcribes the voiceover with faster-whisper to get word-level timestamps.
  4. Downloads a matching vertical stock video (+ a photo for the thumbnail
     background) from Pexels.
  5. Builds a 1920x1080 thumbnail with Pillow.
  6. Assembles the final 1080x1920 video with moviepy, burning word-synced
     captions into the center of the frame.
  7. Uploads the result to YouTube as a Short via the YouTube Data API v3,
     including the custom thumbnail.

--------------------------------------------------------------------------------
INSTALL (Linux, Python 3.10+ recommended):
    pip install g4f requests edge-tts faster-whisper moviepy pillow numpy \
                google-api-python-client google-auth-oauthlib google-auth-httplib2

SYSTEM DEPENDENCIES:
    sudo apt install ffmpeg fonts-dejavu-core

WHERE TO PUT YOUR CREDENTIALS (see the CONFIG block right below the imports):
    - PEXELS_API_KEY      -> https://www.pexels.com/api/                (free)
    - GROQ_API_KEY        -> https://console.groq.com/keys              (free, used only
                              as a fallback if the g4f library fails)
    - client_secrets.json -> Google Cloud Console -> APIs & Services -> Credentials
                              -> Create OAuth client ID -> "Desktop app".
                              Enable the "YouTube Data API v3" for the project first.
                              Save the downloaded JSON next to this script as
                              client_secrets.json.

MEMORY NOTES (target: 16GB RAM host):
    - faster-whisper is restricted to "base" or "small" on CPU with int8
      quantization (~150MB-500MB resident), never "medium"/"large".
    - Captions are rendered as small Pillow PNG frames instead of moviepy's
      ImageMagick-based TextClip, which avoids an extra heavyweight
      dependency and keeps per-frame memory low.
    - moviepy's render uses a fast x264 preset, capped bitrate, and explicit
      clip.close() calls + gc.collect() so buffers are released promptly.
================================================================================
"""

import os
import sys
import re
import json
import gc
import time
import random
import logging
import asyncio
import pickle
import textwrap
from pathlib import Path

import requests
import numpy as np

# ==============================================================================
# CONFIGURATION -- INSERT YOUR API KEYS / FILE PATHS HERE
# ==============================================================================
PEXELS_API_KEY = "PEXELS_API_KEY_HERE"          # https://www.pexels.com/api/
GROQ_API_KEY = "GROQ_API_KEY_HERE"              # https://console.groq.com/keys (g4f fallback)
GROQ_MODEL = "llama-3.3-70b-versatile"                   # any current free Groq chat model

CLIENT_SECRETS_FILE = "client_secrets.json"           # Google OAuth2 Desktop app credentials
YOUTUBE_TOKEN_CACHE = "token.pickle"                   # auto-created after first login
YOUTUBE_PRIVACY_STATUS = "public"                     # "public" | "unlisted" | "private"
YOUTUBE_CATEGORY_ID = "24"                             # 24 = Entertainment

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # adjust for your distro
WHISPER_MODEL_SIZE = "base"                            # "base" or "small" ONLY (RAM budget)
EDGE_TTS_VOICE = "en-US-ChristopherNeural"             # any edge-tts voice name

OUTPUT_DIR = Path("output")
MAX_SHORT_SECONDS = 58                                  # keep comfortably under 60s

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("shorts_bot")


# ==============================================================================
# 1. CONTENT GENERATION (fact, title, description, tags, thumbnail prompt)
# ==============================================================================
def _build_llm_prompt() -> str:
    """Builds the instruction prompt sent to the LLM. Requesting strict JSON
    keeps parsing reliable across different free providers/models."""
    return (
        "You write viral, family-friendly YouTube Shorts about random, surprising "
        "fun facts (science, space, history, animals, or psychology). "
        "Respond with ONLY a valid JSON object, no markdown fences, no commentary, "
        "matching exactly this schema:\n"
        "{\n"
        '  "fact": "A mini-story of 4-6 sentences, about 90-140 words total '
        '(roughly 30-45 seconds when narrated aloud). Pack in extra specific, '
        'surprising details -- context, numbers, a twist -- so it never feels '
        'padded or repetitive. MUST start with the exact phrase '
        '\'Did you know?\' followed by a space.",\n'
        '\'Did you know?\' followed by a space.",\n'
        '  "title": "Catchy YouTube Shorts title, under 90 characters, 1 emoji max",\n'
        '  "description": "2-3 sentence description ending with 5-8 relevant hashtags",\n'
        '  "tags": ["8", "to", "12", "short seo tags", "no hashtags here"],\n'
        '  "topic": "2-4 word visual search term for stock footage, e.g. \'deep ocean\'",\n'
        '  "thumbnail_prompt": "short description of an eye-catching thumbnail concept"\n'
        "}"
    )


def _call_g4f(prompt: str) -> str:
    """Primary LLM call using the free g4f library. Provider/model support in
    g4f shifts often -- if this raises, generate_fact_and_metadata() falls
    back to Groq automatically."""
    import g4f  # imported lazily so the whole script still loads if g4f is absent

    response = g4f.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    # g4f sometimes returns a generator/stream depending on provider; normalize to str
    if not isinstance(response, str):
        response = "".join(response)
    return response


def _call_groq(prompt: str) -> str:
    """Fallback LLM call using Groq's free OpenAI-compatible endpoint."""
    if not GROQ_API_KEY or GROQ_API_KEY == "xxx":
        raise RuntimeError("GROQ_API_KEY is not set")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "max_tokens": 500,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _extract_json(raw_text: str) -> dict:
    """Pulls the first {...} block out of the LLM response and parses it,
    tolerating stray markdown fences some providers add."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {raw_text[:200]!r}")
    return json.loads(match.group(0))


def generate_fact_and_metadata(max_retries: int = 3) -> dict:
    """Generates the fun fact + all video metadata in one LLM call.
    Guarantees the fact begins with the exact phrase 'Did you know?' even if
    the model forgets, by enforcing it in code rather than trusting the LLM."""
    prompt = _build_llm_prompt()
    last_error = None

    for attempt in range(1, max_retries + 1):
        raw = None
        try:
            raw = _call_g4f(prompt)
        except Exception as e:
            log.warning("g4f generation failed (%s); falling back to Groq...", e)
            try:
                raw = _call_groq(prompt)
            except Exception as e2:
                last_error = e2
                log.warning("Groq fallback also failed on attempt %d: %s", attempt, e2)
                continue

        try:
            data = _extract_json(raw)
            required_keys = ["fact", "title", "description", "tags", "topic", "thumbnail_prompt"]
            if not all(k in data for k in required_keys):
                raise ValueError(f"Missing keys in LLM JSON, got: {list(data.keys())}")

            # --- Strict constraint: enforce the "Did you know?" opener ---
            fact = str(data["fact"]).strip()
            if not fact.lower().startswith("did you know?"):
                fact = "Did you know? " + fact.lstrip("? ").strip()
            data["fact"] = fact

            # Basic hygiene on tags/title/description
            data["title"] = str(data["title"]).strip()[:95]
            data["tags"] = [str(t).strip() for t in data["tags"]][:15]
            data["description"] = str(data["description"]).strip()

            log.info("Generated fact: %s", data["fact"])
            return data
        except Exception as e:
            last_error = e
            log.warning("Attempt %d: failed to parse LLM output (%s)", attempt, e)
            time.sleep(1.5)

    raise RuntimeError(f"Could not generate valid content after {max_retries} attempts: {last_error}")


# ==============================================================================
# 2. VOICEOVER (edge-tts) + TRANSCRIPTION (faster-whisper)
# ==============================================================================
async def _edge_tts_save(text: str, output_path: str, voice: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice=voice, rate="+2%")
    await communicate.save(output_path)


def create_voiceover(text: str, output_path: str = None, voice: str = EDGE_TTS_VOICE) -> str:
    """Generates an MP3 voiceover of `text` using Microsoft edge-tts (free, no key needed)."""
    output_path = output_path or str(OUTPUT_DIR / "voiceover.mp3")
    try:
        asyncio.run(_edge_tts_save(text, output_path, voice))
        log.info("Voiceover saved to %s", output_path)
        return output_path
    except Exception as e:
        log.error("edge-tts generation failed: %s", e)
        raise


def transcribe_audio(audio_path: str, model_size: str = WHISPER_MODEL_SIZE) -> list:
    """Transcribes `audio_path` with faster-whisper and returns a flat list of
    {"word", "start", "end"} dicts. Runs on CPU with int8 quantization and
    beam_size=1 to keep peak memory well under 2GB, per the "base"/"small"
    model constraint."""
    if model_size not in ("base", "small"):
        raise ValueError("WHISPER_MODEL_SIZE must be 'base' or 'small' to respect the RAM budget")

    from faster_whisper import WhisperModel

    model = None
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(
            audio_path,
            word_timestamps=True,
            beam_size=1,
            vad_filter=True,
        )
        words = []
        for segment in segments:
            if not segment.words:
                continue
            for w in segment.words:
                words.append({"word": w.word.strip(), "start": w.start, "end": w.end})

        if not words:
            raise RuntimeError("Transcription produced no words -- check the audio file")

        log.info("Transcribed %d words", len(words))
        return words
    except Exception as e:
        log.error("Transcription failed: %s", e)
        raise
    finally:
        # Explicitly release the model before moving on to keep peak RSS low
        del model
        gc.collect()


# ==============================================================================
# 3. VISUAL ASSETS (Pexels stock video/photo + Pillow thumbnail)
# ==============================================================================
def download_pexels_video(query: str, output_path: str = None) -> str:
    """Searches Pexels for a vertical stock video matching `query` and
    downloads the file whose resolution is closest to 1080x1920."""
    if not PEXELS_API_KEY or PEXELS_API_KEY == "xxx":
        raise RuntimeError("PEXELS_API_KEY is not set")

    output_path = output_path or str(OUTPUT_DIR / "background.mp4")
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "orientation": "portrait", "size": "medium", "per_page": 15}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
        if not videos:
            raise ValueError(f"No Pexels videos found for query {query!r}")

        random.shuffle(videos)
        for video in videos:
            files = [f for f in video.get("video_files", []) if (f.get("height") or 0) > (f.get("width") or 0)]
            if not files:
                continue
            # Pick the vertical file closest to Full HD (1080x1920)
            files.sort(key=lambda f: abs((f.get("width") or 0) - 1080))
            best = files[0]
            if (best.get("width") or 0) < 720:
                continue  # skip low-res files

            with requests.get(best["link"], stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(output_path, "wb") as out:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        out.write(chunk)
            log.info("Downloaded stock video to %s", output_path)
            return output_path

        raise ValueError("No suitably vertical video file found in Pexels results")
    except Exception as e:
        log.error("Pexels video download failed: %s", e)
        raise


def download_pexels_photo(query: str, output_path: str = None) -> str:
    """Downloads a landscape stock photo from Pexels to use as the thumbnail background."""
    if not PEXELS_API_KEY or PEXELS_API_KEY == "xxx":
        raise RuntimeError("PEXELS_API_KEY is not set")

    output_path = output_path or str(OUTPUT_DIR / "thumb_bg.jpg")
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "orientation": "landscape", "per_page": 10}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        photos = resp.json().get("photos", [])
        if not photos:
            raise ValueError(f"No Pexels photos found for query {query!r}")

        random.shuffle(photos)
        img_url = photos[0]["src"]["large2x"]
        r = requests.get(img_url, timeout=30)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(r.content)
        log.info("Downloaded thumbnail background to %s", output_path)
        return output_path
    except Exception as e:
        log.error("Pexels photo download failed: %s", e)
        raise


def create_thumbnail(topic_text: str, bg_image_path: str, output_path: str = None,
                      font_path: str = FONT_PATH) -> str:
    """Creates a 1920x1080 thumbnail: darkened stock photo + bold white text
    with a heavy black stroke, auto-shrunk to fit the frame."""
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps

    output_path = output_path or str(OUTPUT_DIR / "thumbnail.jpg")
    try:
        img = Image.open(bg_image_path).convert("RGB")
        img = ImageOps.fit(img, (1920, 1080), Image.LANCZOS)
        img = ImageEnhance.Brightness(img).enhance(0.55)  # darken for text contrast
        img = ImageEnhance.Contrast(img).enhance(1.1)
        draw = ImageDraw.Draw(img)

        text = topic_text.upper().strip()
        stroke_w = 8
        font_size = 150
        lines, line_heights, total_h, max_w = [], [], 0, 0

        while font_size > 40:
            font = ImageFont.truetype(font_path, font_size)
            lines = textwrap.wrap(text, width=max(1, 16 - font_size // 30))
            line_heights = []
            max_w = 0
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_w)
                max_w = max(max_w, bbox[2] - bbox[0])
                line_heights.append(bbox[3] - bbox[1])
            total_h = sum(line_heights) + 24 * (len(lines) - 1)
            if max_w <= 1700 and total_h <= 820:
                break
            font_size -= 6

        y = (1080 - total_h) // 2
        for line, h in zip(lines, line_heights):
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_w)
            w = bbox[2] - bbox[0]
            x = (1920 - w) // 2
            draw.text((x, y), line, font=font, fill="white",
                       stroke_width=stroke_w, stroke_fill="black")
            y += h + 24

        img.save(output_path, quality=95)
        log.info("Thumbnail saved to %s", output_path)
        return output_path
    except Exception as e:
        log.error("Thumbnail creation failed: %s", e)
        raise


# ==============================================================================
# 4. VIDEO ASSEMBLY (moviepy) -- burned-in, word-synced captions
# ==============================================================================
def _group_words(words: list, group_size: int = 3) -> list:
    """Groups whisper words into short caption chunks (2-3 words each reads
    better on Shorts than one word or a full sentence at a time)."""
    groups = []
    for i in range(0, len(words), group_size):
        chunk = words[i:i + group_size]
        text = " ".join(w["word"] for w in chunk).strip()
        if not text:
            continue
        groups.append({"text": text, "start": chunk[0]["start"], "end": chunk[-1]["end"]})
    return groups


def _render_caption_png(text: str, font_path: str = FONT_PATH, font_size: int = 78) -> np.ndarray:
    """Renders one caption chunk to an RGBA numpy array with Pillow (white
    fill, thick black stroke). Avoids depending on moviepy's ImageMagick
    TextClip entirely, which keeps the render both lighter and more portable."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(font_path, font_size)
    dummy = Image.new("RGBA", (10, 10))
    ddraw = ImageDraw.Draw(dummy)
    stroke_w = 6
    bbox = ddraw.textbbox((0, 0), text, font=font, stroke_width=stroke_w)
    pad = 30
    w = (bbox[2] - bbox[0]) + pad * 2
    h = (bbox[3] - bbox[1]) + pad * 2

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill="white",
               stroke_width=stroke_w, stroke_fill="black")
    return np.array(img)


def assemble_video(video_path: str, audio_path: str, words: list,
                    output_path: str = None) -> str:
    """Combines the stock video, voiceover, and burned-in captions into the
    final 1080x1920 @30fps H.264 Short, tuned for low peak memory use."""
    from moviepy import (
        VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips,
    )

    output_path = output_path or str(OUTPUT_DIR / "final_short.mp4")
    audio_clip = bg_clip = final = None
    caption_clips = []

    try:
        audio_clip = AudioFileClip(audio_path)
        duration = min(audio_clip.duration, MAX_SHORT_SECONDS)
        if audio_clip.duration > MAX_SHORT_SECONDS:
            log.warning("Voiceover longer than %ss; trimming video to stay a Short", MAX_SHORT_SECONDS)
            audio_clip = audio_clip.subclipped(0, duration)

        bg_clip = VideoFileClip(video_path, audio=False)
        if bg_clip.duration < duration:
            loops_needed = int(duration // bg_clip.duration) + 1
            bg_clip = concatenate_videoclips([bg_clip] * loops_needed)
        bg_clip = bg_clip.subclipped(0, duration)

        # Resize/crop to a clean 1080x1920 vertical frame
        target_w, target_h = 1080, 1920
        bg_clip = bg_clip.resized(height=target_h)
        if bg_clip.w > target_w:
            bg_clip = bg_clip.cropped(x_center=bg_clip.w / 2, width=target_w)
        elif bg_clip.w < target_w:
            bg_clip = bg_clip.resized(width=target_w).cropped(y_center=bg_clip.h / 2, height=target_h)

        # Build word-synced caption clips, centered on screen
        for group in _group_words(words, group_size=3):
            if group["start"] >= duration:
                break
            clip_end = min(group["end"], duration)
            png = _render_caption_png(group["text"])
            clip = (
                ImageClip(png)
                .with_start(group["start"])
                .with_duration(max(clip_end - group["start"], 0.25))
                .with_position(("center", "center"))
            )
            caption_clips.append(clip)

        final = CompositeVideoClip([bg_clip, *caption_clips], size=(target_w, target_h))
        final = final.with_audio(audio_clip).with_duration(duration)

        final.write_videofile(
            output_path,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="veryfast",     # speed/memory over max compression ratio
            threads=4,
            bitrate="4500k",
            ffmpeg_params=["-crf", "23", "-pix_fmt", "yuv420p"],
            logger=None,           # quiets moviepy's progress bar spam
        )
        log.info("Final video rendered to %s", output_path)
        return output_path
    except Exception as e:
        log.error("Video assembly failed: %s", e)
        raise
    finally:
        # Aggressively release clips/frames -- important on constrained RAM
        for clip in [*caption_clips, bg_clip, audio_clip, final]:
            try:
                if clip is not None:
                    clip.close()
            except Exception:
                pass
        gc.collect()


# ==============================================================================
# 5. YOUTUBE UPLOAD (OAuth2 + Data API v3)
# ==============================================================================
YOUTUBE_UPLOAD_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def confirm_upload(video_path: str, thumbnail_path: str, title: str, description: str) -> bool:
    """Pauses the pipeline so a human can review the rendered video/thumbnail
    before anything goes live. Returns True to proceed with the YouTube
    upload, False to stop the pipeline here (files are left on disk either way)."""
    print("\n" + "=" * 70)
    print("READY TO UPLOAD -- review before it goes live")
    print("=" * 70)
    print(f"Video:       {os.path.abspath(video_path)}")
    print(f"Thumbnail:   {os.path.abspath(thumbnail_path)}")
    print(f"Title:       {title}")
    print(f"Description: {description}")
    print("=" * 70)

    while True:
        answer = input("Publish this video to YouTube now? [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


def _get_authenticated_service():
    """Handles the OAuth2 dance, caching the resulting token so subsequent
    runs don't need a browser login. Expects `client_secrets.json` (a Desktop
    app OAuth client from Google Cloud Console) in the working directory."""
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if os.path.exists(YOUTUBE_TOKEN_CACHE):
        with open(YOUTUBE_TOKEN_CACHE, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CLIENT_SECRETS_FILE):
                raise FileNotFoundError(
                    f"{CLIENT_SECRETS_FILE} not found -- download OAuth Desktop "
                    "credentials from Google Cloud Console and place them here."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, YOUTUBE_UPLOAD_SCOPES)
            creds = flow.run_local_server(port=0)  # opens a browser for the one-time login
        with open(YOUTUBE_TOKEN_CACHE, "wb") as f:
            pickle.dump(creds, f)

    return build("youtube", "v3", credentials=creds)


def upload_to_youtube(video_path: str, thumbnail_path: str, title: str,
                       description: str, tags: list) -> str:
    """Uploads the finished video as a public YouTube Short, then attaches
    the generated thumbnail. Returns the new video's ID."""
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    try:
        youtube = _get_authenticated_service()

        if "#shorts" not in title.lower() and "#shorts" not in description.lower():
            description = description.rstrip() + "\n\n#Shorts"

        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": YOUTUBE_CATEGORY_ID,
            },
            "status": {
                "privacyStatus": YOUTUBE_PRIVACY_STATUS,
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(video_path, chunksize=4 * 1024 * 1024, resumable=True, mimetype="video/mp4")
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                log.info("Upload progress: %d%%", int(status.progress() * 100))

        video_id = response["id"]
        log.info("Uploaded successfully: https://youtu.be/%s", video_id)

        if thumbnail_path and os.path.exists(thumbnail_path):
            try:
                youtube.thumbnails().set(
                    videoId=video_id,
                    media_body=MediaFileUpload(thumbnail_path, mimetype="image/jpeg"),
                ).execute()
                log.info("Custom thumbnail attached")
            except HttpError as e:
                # Custom thumbnails require a verified/phone-linked channel; don't
                # fail the whole pipeline if this specific call is rejected.
                log.warning("Could not set custom thumbnail (channel may need phone verification): %s", e)

        return video_id
    except Exception as e:
        log.error("YouTube upload failed: %s", e)
        raise


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_files = []

    try:
        log.info("Step 1/6: Generating fact + metadata...")
        content = generate_fact_and_metadata()
        fact = content["fact"]
        title = content["title"]
        description = content["description"]
        tags = content["tags"]
        topic = content["topic"]

        log.info("Step 2/6: Generating voiceover...")
        audio_path = create_voiceover(fact)
        temp_files.append(audio_path)

        log.info("Step 3/6: Transcribing for word-level captions...")
        words = transcribe_audio(audio_path)

        log.info("Step 4/6: Fetching stock footage + building thumbnail...")
        video_path = download_pexels_video(topic)
        temp_files.append(video_path)
        photo_path = download_pexels_photo(topic)
        temp_files.append(photo_path)
        thumbnail_path = create_thumbnail(topic, photo_path)

        log.info("Step 5/6: Assembling final video...")
        final_video_path = assemble_video(video_path, audio_path, words)

        if not confirm_upload(final_video_path, thumbnail_path, title, description):
            log.info("Upload cancelled. Review the files and re-run manually if you want to publish:")
            log.info("  Video:     %s", os.path.abspath(final_video_path))
            log.info("  Thumbnail: %s", os.path.abspath(thumbnail_path))
            return

        log.info("Step 6/6: Uploading to YouTube...")
        video_id = upload_to_youtube(final_video_path, thumbnail_path, title, description, tags)

        log.info("Done! https://youtu.be/%s", video_id)

    except Exception as e:
        log.exception("Pipeline failed: %s", e)
        sys.exit(1)
    finally:
        for f in temp_files:
            try:
                if f and os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass


if __name__ == "__main__":
    main()
