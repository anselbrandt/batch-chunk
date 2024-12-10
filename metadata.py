def datesDict(file):
    dates = {
        date: episode
        for date, episode in [
            (line.split(" - ")[0], line.split(" - ")[1])
            for line in open(file).read().splitlines()
        ]
    }
    return dates


def titlesDict(file):
    titles = {
        title: episode
        for title, episode in [
            (line.split(" - ")[0], line.split(" - ")[1])
            for line in open(file).read().splitlines()
        ]
    }
    return titles


def omnibusMeta(file):
    metadata = [
        tuple(episode.splitlines()) for episode in open(file).read().split("\n\n")
    ]
    episodes = {
        episode: {
            "episode": episode,
            "showpage": showpage,
            "title": title,
            "date": date,
            "desc": desc,
            "link": link,
        }
        for episode, showpage, title, date, desc, link in metadata
    }

    dates = {episode: date for episode, showpage, title, date, desc, link in metadata}
    titles = {episode: title for episode, showpage, title, date, desc, link in metadata}

    return {"episodes": episodes, "dates": dates, "titles": titles}
