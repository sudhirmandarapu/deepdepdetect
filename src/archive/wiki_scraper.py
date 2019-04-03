import os
import requests
from bs4 import BeautifulSoup


def scrape_antidepressants():
    page = requests.get('https://en.wikipedia.org/wiki/List_of_antidepressants')

    soup = BeautifulSoup(page.content, 'html.parser')
    li_items = soup.find_all('li')

    names = []
    for item in li_items:
        if item.get('id') is None:
            if item.string is not None:
                if len(item.string) > 1:
                    names.append(str(item.string))
            else:
                link = item.find('a')
                if link is not None:
                    names.append((str(link.contents[0].string)))
                    if len(item.contents) > 1:
                        contents = item.contents[1]
                        try:
                            alt_names = contents[contents.index('(')+1:contents.index(')')].split(', ')
                            names += alt_names
                        except ValueError:
                            pass
    return names


def format_words():
    file_location = os.getenv('INPUT_FILE')
    output = os.getenv('UNIQUE_WORDS_FILE')
    f = open(file_location, 'r')
    o = open(output, 'w+')
    contents = f.read().split('\n')
    found = {}
    uniques = []
    i = 0
    while i < len(contents):
        word = contents[i].split(" ")[0]
        if word not in found:
            uniques.append(word)
            found[word] = True
        i += 1
    uniques.sort()
    for word in uniques:
        o.write(word+'\n')
    f.close()
    o.close()


format_words()
