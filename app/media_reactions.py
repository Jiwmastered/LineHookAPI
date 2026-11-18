import random
import re
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class StickerItem:
    package_id: str
    sticker_id: str


@dataclass(frozen=True)
class GifItem:
    key: str
    title: str
    mp4_url: str
    preview_url: str


# Official LINE Messaging API default animated & special sticker packages:
# - 11537: Brown & Cony & Sally Animated Special (52002734 - 52002773)
# - 11538: CHOCO & Friends Animated Special (51626494 - 51626533)
# - 446:   Moon Special Edition (1988 - 2027)
STICKER_MOODS: dict[str, list[StickerItem]] = {
    "happy": [
        StickerItem("11537", "52002734"),
        StickerItem("11537", "52002735"),
        StickerItem("446", "1988"),
    ],
    "laugh": [
        StickerItem("11537", "52002740"),
        StickerItem("446", "1990"),
        StickerItem("11538", "51626501"),
    ],
    "love": [
        StickerItem("11537", "52002736"),
        StickerItem("11538", "51626495"),
        StickerItem("446", "1992"),
    ],
    "ok": [
        StickerItem("11537", "52002738"),
        StickerItem("11538", "51626494"),
        StickerItem("446", "1989"),
    ],
    "celebrate": [
        StickerItem("11537", "52002741"),
        StickerItem("11537", "52002745"),
        StickerItem("11538", "51626500"),
    ],
    "cool": [
        StickerItem("11537", "52002744"),
        StickerItem("446", "1995"),
    ],
    "shock": [
        StickerItem("11537", "52002748"),
        StickerItem("11538", "51626518"),
        StickerItem("446", "2006"),
    ],
    "sad": [
        StickerItem("11537", "52002750"),
        StickerItem("11538", "51626520"),
        StickerItem("446", "2010"),
    ],
    "cry": [
        StickerItem("11537", "52002751"),
        StickerItem("11538", "51626521"),
        StickerItem("446", "2011"),
    ],
    "angry": [
        StickerItem("11537", "52002752"),
        StickerItem("11538", "51626522"),
        StickerItem("446", "2013"),
    ],
    "tired": [
        StickerItem("11537", "52002755"),
        StickerItem("11538", "51626525"),
        StickerItem("446", "2015"),
    ],
    "sorry": [
        StickerItem("11537", "52002759"),
        StickerItem("11538", "51626528"),
    ],
    "thinking": [
        StickerItem("11537", "52002746"),
        StickerItem("446", "1998"),
    ],
}

# Verified HTTPS MP4 + JPEG preview URLs for LINE Video Message animated GIFs
HILARIOUS_GIFS: dict[str, GifItem] = {
    "laugh": GifItem(
        key="laugh",
        title="Bursting out laughing",
        mp4_url="https://media.giphy.com/media/10JhviFuU2gWD6/giphy.mp4",
        preview_url="https://media.giphy.com/media/10JhviFuU2gWD6/480w_s.jpg",
    ),
    "mindblown": GifItem(
        key="mindblown",
        title="Mind blown",
        mp4_url="https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.mp4",
        preview_url="https://media.giphy.com/media/26ufdipQqU2lhNA4g/480w_s.jpg",
    ),
    "facepalm": GifItem(
        key="facepalm",
        title="Epic facepalm",
        mp4_url="https://media.giphy.com/media/TJawtKM6OCKkvwCIqX/giphy.mp4",
        preview_url="https://media.giphy.com/media/TJawtKM6OCKkvwCIqX/480w_s.jpg",
    ),
    "dance": GifItem(
        key="dance",
        title="Victory dance",
        mp4_url="https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.mp4",
        preview_url="https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/480w_s.jpg",
    ),
    "cat_typing": GifItem(
        key="cat_typing",
        title="Cat furiously typing",
        mp4_url="https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.mp4",
        preview_url="https://media.giphy.com/media/JIX9t2j0ZTN9S/480w_s.jpg",
    ),
    "popcorn": GifItem(
        key="popcorn",
        title="Grabbing the popcorn",
        mp4_url="https://media.giphy.com/media/guufsF0Az3Lpu/giphy.mp4",
        preview_url="https://media.giphy.com/media/guufsF0Az3Lpu/480w_s.jpg",
    ),
    "shocked": GifItem(
        key="shocked",
        title="Total shock",
        mp4_url="https://media.giphy.com/media/3o72F8t9TDi2xVnxOE/giphy.mp4",
        preview_url="https://media.giphy.com/media/3o72F8t9TDi2xVnxOE/480w_s.jpg",
    ),
    "thumbsup": GifItem(
        key="thumbsup",
        title="Approved thumbs up",
        mp4_url="https://media.giphy.com/media/111ebonMs90YLu/giphy.mp4",
        preview_url="https://media.giphy.com/media/111ebonMs90YLu/480w_s.jpg",
    ),
    "clap": GifItem(
        key="clap",
        title="Standing ovation",
        mp4_url="https://media.giphy.com/media/nbvFVPiEiJH6JOGIok/giphy.mp4",
        preview_url="https://media.giphy.com/media/nbvFVPiEiJH6JOGIok/480w_s.jpg",
    ),
    "confused": GifItem(
        key="confused",
        title="Deeply confused math",
        mp4_url="https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.mp4",
        preview_url="https://media.giphy.com/media/3o7btPCcdNniyf0ArS/480w_s.jpg",
    ),
}

MEDIA_TAG_PATTERN = re.compile(r"\[(STICKER|GIF)\s*:\s*([a-zA-Z0-9_\- ]+)\]", flags=re.IGNORECASE)

GIF_KEYWORD_ALIASES: dict[str, str] = {
    "funny": "laugh",
    "lol": "laugh",
    "haha": "laugh",
    "hilarious": "laugh",
    "ตลก": "laugh",
    "ขำ": "laugh",
    "ฮา": "laugh",
    "wow": "mindblown",
    "boom": "mindblown",
    "wtf": "confused",
    "what": "confused",
    "งง": "confused",
    "cat": "cat_typing",
    "coding": "cat_typing",
    "work": "cat_typing",
    "แมว": "cat_typing",
    "drama": "popcorn",
    "party": "dance",
    "happy": "dance",
    "yay": "dance",
    "good": "thumbsup",
    "nice": "thumbsup",
    "ok": "thumbsup",
    "bravo": "clap",
    "congrats": "clap",
    "omg": "shocked",
    "shock": "shocked",
    "fail": "facepalm",
    "sigh": "facepalm",
}

STICKER_KEYWORD_ALIASES: dict[str, str] = {
    "haha": "laugh",
    "lol": "laugh",
    "funny": "laugh",
    "ตลก": "laugh",
    "ขำ": "laugh",
    "heart": "love",
    "รัก": "love",
    "thanks": "happy",
    "thank": "happy",
    "ขอบคุณ": "happy",
    "yay": "celebrate",
    "party": "celebrate",
    "congrats": "celebrate",
    "sleepy": "tired",
    "ง่วง": "tired",
    "เหนื่อย": "tired",
    "omg": "shock",
    "wow": "shock",
    "ตกใจ": "shock",
    "mad": "angry",
    "โกรธ": "angry",
    "เศร้า": "sad",
    "ร้องไห้": "cry",
    "ขอโทษ": "sorry",
}

MEDIA_PROMPT_INSTRUCTION = (
    "You can optionally include ONE reaction tag at the very end of your reply when it fits the mood "
    "or when the user asks for a sticker or funny GIF:\n"
    "- [STICKER:<mood>] where <mood> is one of: "
    + ", ".join(STICKER_MOODS.keys())
    + "\n- [GIF:<mood>] where <mood> is one of: "
    + ", ".join(HILARIOUS_GIFS.keys())
    + "\nOnly include a tag when it genuinely adds humor or emotion, or when requested. Never explain the tag."
)


def build_sticker_message(mood: str = "") -> dict[str, Any]:
    """Return a valid LINE sticker message object matching the requested mood."""
    normalized = mood.strip().lower().replace(" ", "_")
    normalized = STICKER_KEYWORD_ALIASES.get(normalized, normalized)
    options = STICKER_MOODS.get(normalized)
    if not options:
        # Try partial keyword match
        for alias, target in STICKER_KEYWORD_ALIASES.items():
            if alias in normalized:
                options = STICKER_MOODS.get(target)
                break
    if not options:
        options = random.choice(list(STICKER_MOODS.values()))
    chosen = random.choice(options)
    return {
        "type": "sticker",
        "packageId": chosen.package_id,
        "stickerId": chosen.sticker_id,
    }


def build_gif_video_message(mood: str = "") -> tuple[GifItem, dict[str, Any]]:
    """Return a LINE video message object for an animated hilarious GIF (MP4 + JPEG preview)."""
    normalized = mood.strip().lower().replace(" ", "_")
    normalized = GIF_KEYWORD_ALIASES.get(normalized, normalized)
    item = HILARIOUS_GIFS.get(normalized)
    if item is None:
        for key in HILARIOUS_GIFS:
            if key in normalized:
                item = HILARIOUS_GIFS[key]
                break
    if item is None:
        for alias, target in GIF_KEYWORD_ALIASES.items():
            if alias in normalized:
                item = HILARIOUS_GIFS.get(target)
                break
    if item is None:
        item = random.choice(list(HILARIOUS_GIFS.values()))

    return item, {
        "type": "video",
        "originalContentUrl": item.mp4_url,
        "previewImageUrl": item.preview_url,
    }


async def search_giphy_gif(query: str, api_key: str) -> dict[str, Any] | None:
    """Optionally query Giphy's live search API when GIPHY_API_KEY is configured."""
    if not api_key.strip() or not query.strip():
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.giphy.com/v1/gifs/search",
                params={"api_key": api_key.strip(), "q": query.strip(), "limit": 5, "rating": "pg-13"},
            )
            if resp.status_code != 200:
                return None
            items = resp.json().get("data") or []
            random.shuffle(items)
            for item in items:
                images = item.get("images") or {}
                mp4_url = (images.get("original_mp4") or {}).get("mp4") or (
                    images.get("downsized_small") or {}
                ).get("mp4")
                gid = item.get("id")
                if mp4_url and gid:
                    preview_url = f"https://media.giphy.com/media/{gid}/480w_s.jpg"
                    return {
                        "type": "video",
                        "originalContentUrl": mp4_url,
                        "previewImageUrl": preview_url,
                    }
    except Exception:
        return None
    return None


def extract_media_tags(reply_text: str) -> tuple[str, list[tuple[str, str]]]:
    """Extract and strip [STICKER:...] / [GIF:...] tags from an AI reply."""
    found: list[tuple[str, str]] = []
    for match in MEDIA_TAG_PATTERN.finditer(reply_text):
        kind = match.group(1).upper()
        arg = match.group(2).strip()
        found.append((kind, arg))

    cleaned = MEDIA_TAG_PATTERN.sub("", reply_text).strip()
    return cleaned, found


def detect_user_media_request(user_text: str) -> str | None:
    """Detect if the user explicitly asked for a GIF/meme or sticker in English or Thai."""
    lower = user_text.lower()
    if any(k in lower for k in ("gif", "meme", "กิ๊ฟ", "มีม")):
        return "GIF"
    if any(k in lower for k in ("sticker", "สติกเกอร์", "สติ๊กเกอร์")):
        return "STICKER"
    return None
