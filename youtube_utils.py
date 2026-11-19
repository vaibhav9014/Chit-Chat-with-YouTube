"""
YouTube helper functions.

Uses:
- YouTube Data API v3 for video title, channel, thumbnail, and availability
- Public thumbnail URL as fallback if no API key is available
"""

import re
import requests


YOUTUBE_ID_REGEX = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|embed/|v/|shorts/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})"
)

BARE_VIDEO_ID_REGEX = re.compile(
    r"^[A-Za-z0-9_-]{11}$"
)


def extract_video_id(text: str):
    """
    Extracts a YouTube video ID from a normal link, Shorts link,
    youtu.be link, embed link, or a bare video ID.
    """

    if not text:
        return None

    text = text.strip()

    if BARE_VIDEO_ID_REGEX.match(text):
        return text

    match = YOUTUBE_ID_REGEX.search(text)

    if match:
        return match.group(1)

    return None


def get_video_metadata(video_id: str, youtube_api_key: str = ""):
    """
    Gets video title, channel name, and thumbnail.
    """

    fallback = {
        "title": "YouTube video",
        "channel": "",
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
    }

    if not youtube_api_key:
        return fallback

    url = "https://www.googleapis.com/youtube/v3/videos"

    params = {
        "part": "snippet",
        "id": video_id,
        "key": youtube_api_key,
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=8,
        )

        response.raise_for_status()

        data = response.json()

        items = data.get("items", [])

        if not items:
            return fallback

        snippet = items[0]["snippet"]

        return {
            "title": snippet.get("title", "YouTube video"),
            "channel": snippet.get("channelTitle", ""),
            "thumbnail": snippet.get(
                "thumbnails", {}
            ).get(
                "high", {}
            ).get(
                "url",
                f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
            ),
        }

    except requests.RequestException:
        return fallback