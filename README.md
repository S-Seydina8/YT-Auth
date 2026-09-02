# YT Auth — YouTube Shorts Automation Pipeline

A single-file Python pipeline that generates a complete YouTube Short from nothing:
LLM writes the script → TTS narrates it → Whisper timestamps it → stock footage and
CC0 music are fetched → moviepy burns in word-synced captions → the result is uploaded
to YouTube via the Data API v3, thumbnail included.

Runs locally on Linux. Tuned for a 16 GB RAM host. Every external service used is on a
free tier.

---

## 1. Repository layout

```
YT Auth/
└── files/                             ← the actual working directory
    ├── yt_shorts_automation_V2.py     ← CURRENT pipeline (37 KB)
    ├── youtube_shorts_automation.py   ← V1, superseded (29 KB)
    ├── requirements.txt               ← 9 deps, unpinned
    ├── client_secrets.json            ← Google OAuth2 "Desktop app" credentials
    ├── token.pickle                   ← cached OAuth token (auto-created, pickled)
    ├── run_youtube_automation.sh       ← launcher: activates venv, picks newest script
    ├── Start YouTube Automation.desktop ← double-clickable GNOME entry → the .sh
    ├── run youtube automation.txt      ← one-line note holding the run command
    ├── venv/                          ← 693 MB virtualenv
    ├── __pycache__/
    └── output/
        ├── final_short.mp4            ← last render (19 MB)
        └── thumbnail.jpg              ← last thumbnail (326 KB)
```

**Note on the structure:** `files/` is the real project root — the launcher, the venv, and
all relative paths (`output/`, `client_secrets.json`, `token.pickle`) resolve from there.

---

## 2. Pipeline stages

`main()` in `yt_shorts_automation_V2.py` runs six steps, logging each as `Step N/6`:

| Step | Function | Service | Output |
|---|---|---|---|
| 1 | `generate_fact_and_metadata()` | g4f → Groq fallback | fact, title, description, tags, topic, thumbnail prompt (strict JSON) |
| 2 | `create_voiceover()` | edge-tts (`en-US-ChristopherNeural`, rate +2%) | `output/voiceover.mp3` |
| 3 | `transcribe_audio()` | faster-whisper `base`, CPU, int8 | flat list of `{word, start, end}` |
| 4 | `download_pexels_video()` + `download_pexels_photo()` + `create_thumbnail()` | Pexels, Pillow | `background.mp4`, `thumb_bg.jpg`, `thumbnail.jpg` (1920×1080) |
| 5 | `assemble_video()` (+ `download_background_music()`) | Freesound, moviepy/ffmpeg | `output/final_short.mp4` (1080×1920, 30 fps, H.264) |
| — | `confirm_upload()` | **interactive y/n gate** | prints paths + metadata, waits for a human |
| 6 | `upload_to_youtube()` | YouTube Data API v3 | video ID; `youtube.thumbnails().set()` afterwards |

On success, `finally` deletes the intermediates listed in `temp_files` (voiceover,
background video, thumbnail source photo). The final MP4 and thumbnail are kept.

### Design decisions worth knowing

- **Captions are Pillow-rendered PNGs**, not moviepy `TextClip`. This drops the
  ImageMagick dependency entirely and keeps per-frame memory low. Words are grouped
  3 at a time (`_group_words`) — reads better on Shorts than one word or a full sentence.
- **Whisper is hard-capped to `base`/`small`.** `transcribe_audio()` raises on anything
  else. `medium`/`large` would blow the RAM budget; the model is explicitly `del`'d and
  `gc.collect()`'d in a `finally`.
- **Memory hygiene throughout `assemble_video()`**: every clip is `.close()`'d in a
  `finally`, the music temp file is unlinked, `gc.collect()` runs at the end. Render uses
  `preset=veryfast`, `bitrate=4500k`, `crf 23`, 4 threads.
- **Music is filtered to `license:"Creative Commons 0"`** at the Freesound query level, so
  results are public-domain and carry no attribution obligation. Mixed at 12% volume with
  a 1 s fade-in / 1.5 s fade-out. If the fetch or mix fails, it logs a warning and
  continues voice-only rather than aborting.
- **Anti-templating:** `HOOK_STYLES` (6 entries) and `EDITORIAL_TONES` (5) are sampled per
  run and injected into the prompt, and the prompt explicitly bans "Did you know" /
  "Believe it or not" openers. Duration is randomized 30–58 s and converted to a target
  word count at ~2.3–2.9 words/sec.
- **moviepy 2.x API.** The script uses the immutable-style methods (`with_start`,
  `subclipped`, `resized`, `cropped`, `with_audio`, `with_position`) and there's a comment
  documenting the rename. `from moviepy.editor import ...` will not work here.

---

## 3. Configuration surface

All of it is a single block at the top of `yt_shorts_automation_V2.py` (lines 65–86):

| Constant | Current value | Notes |
|---|---|---|
| `PEXELS_API_KEY` | *hardcoded* | free, pexels.com/api |
| `GROQ_API_KEY` | *hardcoded* | free, console.groq.com/keys — fallback only |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | replaced `llama-3.1-8b-instant` (retired) |
| `FREESOUND_API_KEY` | *hardcoded* | free, freesound.org/apiv2/apply |
| `BACKGROUND_MUSIC_VOLUME` | `0.12` | kept low so it never masks the voice |
| `CLIENT_SECRETS_FILE` | `client_secrets.json` | Google OAuth2 Desktop app |
| `YOUTUBE_TOKEN_CACHE` | `token.pickle` | auto-created on first login |
| `YOUTUBE_PRIVACY_STATUS` | **`public`** | ⚠️ see §7 |
| `YOUTUBE_CATEGORY_ID` | `24` | Entertainment |
| `FONT_PATH` | DejaVuSans-Bold | distro-specific |
| `WHISPER_MODEL_SIZE` | `base` | `base` or `small` only |
| `EDGE_TTS_VOICE` | `en-US-ChristopherNeural` | any edge-tts voice |
| `TARGET_DURATION_MIN/MAX_SECONDS` | `30` / `58` | randomized per run |
| `MAX_SHORT_SECONDS` | `60` | hard trim if the model overshoots |

Groq is called with `reasoning_effort="low"`, `reasoning_format="hidden"`, and
`max_completion_tokens` (not `max_tokens`) — `gpt-oss-20b` is a reasoning model, and
without those three the hidden reasoning eats the whole budget and returns empty
`content`. There's an explicit guard that raises on an empty response.

OAuth scope is minimal: `["https://www.googleapis.com/auth/youtube.upload"]`.

---

## 4. V1 vs V2

`youtube_shorts_automation.py` is superseded. V2 adds:

- **Background music** — `FREESOUND_API_KEY`, `BACKGROUND_MUSIC_VOLUME`,
  `RELAXING_MUSIC_QUERIES`, `download_background_music()`, and the `CompositeAudioClip`
  mix in `assemble_video()`. None of this exists in V1.
- **Randomized duration** — V1 had a fixed prompt and `MAX_SHORT_SECONDS = 58`. V2 threads
  a per-run `target_seconds` into `_build_llm_prompt()` to vary length and word count.
- **Hook/tone rotation** — `HOOK_STYLES` and `EDITORIAL_TONES` are V2-only. V1's prompt
  was static, so every V1 script read the same.
- **Groq model + reasoning params** — V1 used `llama-3.3-70b-versatile` with plain
  `max_tokens`. V2 moved to `openai/gpt-oss-20b` with the reasoning-model parameters.

The launcher prefers V2 automatically (`for candidate in "yt_shorts_automation_V2.py"
"youtube_shorts_automation.py"`), so V1 is dead code unless V2 is deleted. `files.zip`
contains only V1 and is a stale snapshot from 2026-08-14.

---

## 5. Install and run

**System packages:**
```bash
sudo apt install ffmpeg fonts-dejavu-core
```

**Python (3.10+; the venv here is 3.13):**
```bash
cd "files"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Google Cloud setup (one time):**
1. Create a project → enable **YouTube Data API v3**.
2. APIs & Services → Credentials → Create OAuth client ID → **Desktop app**.
3. Save the JSON as `files/client_secrets.json`.
4. Add your Google account as a **test user** on the OAuth consent screen while the app is
   in Testing, or the flow will refuse to issue a token.

**Run:**
```bash
cd "files" && ./run_youtube_automation.sh
```
or double-click `Start YouTube Automation.desktop`. First run opens a browser for the
OAuth consent; the token is cached to `token.pickle` afterwards.

The launcher `cd`s to its own directory (double-clicking gives an unpredictable cwd),
probes `venv/ .venv/ env/ ../venv/` for an activate script, picks the newest pipeline
script, and holds the terminal open with `read -p` so you can read errors.

---

## 6. Cost and rate limits

Everything is free-tier, but **the YouTube API quota is the binding constraint**:

- Default quota: **10,000 units/day** per Cloud project.
- `videos.insert` costs **1,600 units**; `thumbnails.set` costs **50**.
- → **~6 uploads/day maximum**, and the quota resets at midnight **Pacific Time**, not on
  a rolling window.

Verify current numbers at
[determine_quota_cost](https://developers.google.com/youtube/v3/determine_quota_cost) and
in Cloud Console → APIs & Services → YouTube Data API v3 → Quotas. *(These figures are
from prior knowledge — I could not confirm them against a live source in this session.)*

Other limits: Pexels 200 req/hour, 20k/month. Groq free tier is rate-limited per model.
Freesound token auth covers preview-quality MP3s only (which is all this uses — no OAuth2
needed). edge-tts and g4f are unofficial and have no published limits or guarantees.

---

## 7. Known issues

Ordered by how much they can cost you.

### Critical — credentials

Live secrets sit in plaintext in **three** places: inline in both `.py` files,
`backup/APIs Keys` + `backup/client_secrets.json`, and inside `files.zip` (which bundles
V1 *with its keys baked in*). Permissions are inconsistent — `youtube_shorts_automation.py`
is `600`, but `yt_shorts_automation_V2.py`, both `client_secrets.json` copies,
`token.pickle`, `backup/*`, and `files.zip` are all **`664`, group- and world-readable**.
The file with the keys in it is the readable one.

Consequence: sharing the folder, the zip, or a screenshot of the config block leaks all
four credentials. Anyone with the OAuth client + a cached token can upload to your
channel. **These keys should be treated as already exposed and rotated.**

### High — `pickle.load` on a group-writable file

`_get_authenticated_service()` does `pickle.load(open("token.pickle", "rb"))`.
Unpickling executes arbitrary code by design, and that file is mode `664`. Any process
that can write it gains code execution as your user on the next run. `google.oauth2`
supports JSON token storage (`Credentials.to_json()` /
`Credentials.from_authorized_user_file()`) — there is no reason to pickle credentials.

### Medium — a revoked token bricks the pipeline

```python
if creds and creds.expired and creds.refresh_token:
    creds.refresh(Request())      # ← no try/except
```
If the refresh token is revoked or expired (Google expires them for apps left in
*Testing* status), `refresh()` raises and propagates. The interactive fallback right below
is never reached. Fix is a `try/except` around the refresh that falls through to
`InstalledAppFlow`, or you'll be deleting `token.pickle` by hand.

### Medium — unpinned dependencies

`requirements.txt` has zero version constraints. This already bit the project once: the
moviepy 2.x rename of `set_start`→`with_start` etc. is documented in a code comment. A
fresh `pip install -r requirements.txt` can break the script on any upstream release.
Pin with `pip freeze > requirements.txt` from the working venv.

### Medium — no repetition guard

Nothing records what's already been published. The LLM can generate the same fun fact
twice, and hook/tone rotation only varies *phrasing*, not subject matter. A JSON history
file of past facts, fed into the prompt as a "don't repeat these" list, would close this.

### Low — code details

- **`create_thumbnail()`**: `font` is referenced after the `while font_size > 40` loop but
  only bound inside it. Safe today only because `font_size` starts at 150; lowering that
  initial value to ≤40 turns it into a `NameError`.
- **`main()`'s `finally`** deletes `temp_files` even when you answer **n** at the confirm
  gate. Declining an upload therefore discards the voiceover, background video, and
  thumbnail source — re-running regenerates them and re-spends API calls.
- **`_render_caption_png()`** sizes the canvas from `bbox[3] - bbox[1]`, discarding the
  top offset. The 30 px pad absorbs it for DejaVu, but a font with deeper descenders
  could clip.
- **693 MB `venv/` lives inside the project.** It must never enter version control or a
  backup archive. A `.gitignore` is included at the repo root for when you `git init`.

---

## 8. What to avoid

**Credentials**
- Don't share this folder, `files.zip`, `backup/`, or a screenshot of the config block —
  each leaks all four secrets.
- Don't `git init` and commit before the keys are externalized to `.env` and `.gitignore`
  is in place. Git history is forever; a committed key is leaked even after you delete it.
- Don't rotate one key and leave the duplicates — fix all three locations or you'll paste
  a dead key back in from `backup/` later.

**Publishing**
- Don't leave `YOUTUBE_PRIVACY_STATUS = "public"` while testing. Use `"private"` until a
  full run is verified end-to-end.
- Don't remove `confirm_upload()`, and don't put this on a cron job or systemd timer. That
  human gate is the only thing between a bad LLM output and your public channel. Fully
  unattended publishing is the single highest-risk change you could make here.
- Don't assume `selfDeclaredMadeForKids: False` is a formality. It's a COPPA declaration
  and it must stay accurate for the content you actually upload.
- Don't skip YouTube's synthetic-content disclosure where it applies. This is
  AI-scripted, AI-narrated content; check the current
  [monetization policies](https://support.google.com/youtube/answer/1311392) and the
  altered-content disclosure rules before publishing at any volume. *(Policy specifics
  change often — verify directly, don't rely on this file.)*
- Don't chase volume. The quota caps you at ~6/day regardless, and high-volume templated
  output is exactly the pattern YouTube's inauthentic-content rules target. Six good
  videos beat sixty templated ones.

**Dependencies**
- Don't build anything you depend on around `g4f`. It reverse-engineers other providers'
  endpoints, routes your prompts through arbitrary third-party hosts, breaks constantly,
  and sits in a licensing grey area. The Groq fallback is the reliable path — consider
  inverting the order so Groq is primary and g4f is the fallback.
- Don't upgrade moviepy, Pillow, or faster-whisper without testing a full render. Pin
  first.
- Don't switch Whisper to `medium`/`large` on this host. The guard in `transcribe_audio()`
  exists for a reason.

**Assets**
- Don't drop the Freesound `license:"Creative Commons 0"` filter. It's the only thing
  keeping the audio free of attribution and copyright-claim obligations. Freesound is
  user-uploaded and other licenses there *do* require credit.
- Don't resell Pexels footage as a standalone asset. Using it inside a video is fine
  under their license; redistributing the clip itself is not.

**Housekeeping**
- Don't keep `files.zip` — it's a stale V1 snapshot whose only distinguishing content is
  live credentials.
- Don't let `output/` accumulate. Each run leaves ~19 MB.
- Don't back up `venv/`. Recreate it from `requirements.txt` instead.

---

## 9. Recommended next steps

In priority order:

1. **Rotate all four credentials** (Pexels, Groq, Freesound, Google OAuth client) — treat
   them as compromised.
2. **Move secrets to `.env`**, load with `os.environ`, and `chmod 600 .env`. See
   `.env.example` at the repo root.
3. **`chmod 600`** `client_secrets.json` and `token.pickle`; delete `backup/` and
   `files.zip` once §1 and §2 are done.
4. **Replace pickle with JSON** token storage.
5. **Wrap `creds.refresh()`** in a `try/except` that falls back to the interactive flow.
6. **Pin `requirements.txt`** from the working venv.
7. **Add a published-facts history file** to prevent repeats.
