import os
import logging

import chromadb
from chromadb.utils import embedding_functions

from utils import getFiles
from metadata import datesDict, titlesDict, omnibusMeta

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("logs.txt"), stream_handler],
)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2", device="cuda"
)

chroma_client = chromadb.PersistentClient(path="/home/ansel/chromadb")

try:
    collection = chroma_client.get_collection(
        name="roderick", embedding_function=sentence_transformer_ef
    )
except:
    collection = chroma_client.create_collection(
        name="roderick", embedding_function=sentence_transformer_ef
    )

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

chunkedDir = os.path.join(ROOT, "chunked")

import sqlite3


import sqlite3


def getChunks(filepath, show, episode):
    conn = sqlite3.connect("transcripts.db")
    c = conn.cursor()
    c.execute(
        """SELECT idx, wavefilepath FROM lines WHERE showname LIKE ? AND episode LIKE ?""",
        (show, episode),
    )
    results = c.fetchall()
    conn.commit()
    conn.close()
    wav_dict = {str(idx): filepath for idx, filepath in results}
    file = open(filepath).read().split("\n\n")
    all_speech = []
    all_wavfiles = []
    for chunk in file:
        lines = [
            (
                line.split("|")[0],
                line.split("|")[1].split(": ")[0],
                line.split("|")[1].split(": ")[1].strip(),
            )
            for line in chunk.split("\n")
        ]
        wavfiles = [f"{speaker}|{wav_dict[idx]}" for idx, speaker, speech in lines]

        chunk_speech = " ".join([speech for idx, speaker, speech in lines])
        all_speech.append(chunk_speech)
        all_wavfiles.append(wavfiles)
    return (all_wavfiles, all_speech)


chunked_files = getFiles(chunkedDir)

hosts = {
    "omnibus": ["John Roderick", "Ken Jennings"],
    "roadwork": ["John Roderick", "Dan Benjamin"],
    "rotl": ["John Roderidk", "Merlin Mann"],
}

for filepath, show, filename in chunked_files:
    episode = filename.split("_-_")[0] if "_-_" in filename else filename.split(".")[0]
    date = dates[show][episode]
    title = titles[show][episode]
    wavfiles, chunks = getChunks(filepath, show, episode)
    documents = chunks
    metadatas = [
        {
            "podcast": show,
            "hosts": ",".join(hosts[show]),
            "episode": episode,
            "title": title,
            "date": date,
            "wavfiles": ",".join(wavfiles[i]),
        }
        for i, chunk in enumerate(chunks)
    ]
    ids = [f"{show}_{episode}_{i}" for i, chunk in enumerate(chunks)]
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    logging.info(f"{show}-{episode}-{title}")
