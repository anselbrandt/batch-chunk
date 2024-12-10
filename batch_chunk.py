import os
import logging
from dotenv import load_dotenv
from semantic_router.encoders import OpenAIEncoder
from semantic_router.splitters import RollingWindowSplitter
from semantic_router.utils.logger import logger
from utils import getTranscriptFiles, srt_to_transcript
from metadata import datesDict, titlesDict, omnibusMeta

load_dotenv()

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("logs.txt"), stream_handler],
)

logger.setLevel("WARNING")  # reduce logs from splitter

encoder = OpenAIEncoder(name="text-embedding-3-small")


rotl_titles = titlesDict("rotl_titles.txt")
rotl_dates = datesDict("rotl_dates.txt")
roadwork_titles = titlesDict("roadwork_titles.txt")
roadwork_dates = datesDict("roadwork_dates.txt")
omnibus = omnibusMeta("omnibus_metadata.txt")
dates = {"rotl": rotl_dates, "roadwork": roadwork_dates, "omnibus": omnibus["dates"]}
titles = {
    "rotl": rotl_titles,
    "roadwork": roadwork_titles,
    "omnibus": omnibus["titles"],
}

ROOT = os.getcwd()
transcriptDir = os.path.join(ROOT, "clean_labeled")
transcriptFiles = getTranscriptFiles(transcriptDir)

chunkedDir = os.path.join(ROOT, "chunked")
os.makedirs(chunkedDir, exist_ok=True)

for filepath, show, filename in transcriptFiles:
    os.makedirs(os.path.join(chunkedDir, show), exist_ok=True)
    episode = filename.split("_-_")[0] if "_-_" in filename else filename.split(".")[0]
    date = dates[show][episode]
    title = titles[show][episode]
    transcript = srt_to_transcript(filepath)
    content_with_speaker = [
        f"{idx}|{speaker}: {speech} " for idx, start, end, speaker, speech in transcript
    ]
    splitter = RollingWindowSplitter(
        encoder=encoder,
        dynamic_threshold=True,
        min_split_tokens=100,
        max_split_tokens=500,
        window_size=2,
        plot_splits=False,  # set this to true to visualize chunking
        enable_statistics=False,  # to print chunking stats
    )

    splits = splitter(content_with_speaker)
    chunks = ["\n".join(split.docs) for split in splits]
    text = "\n\n".join(chunks)
    outpath = os.path.join(ROOT, "chunked", show, filename.replace(".srt", ".txt"))
    f = open(outpath, "w")
    f.write(text)
    f.close()
    logging.info(f"{show}|{filename}")