import sys
import os
from pathlib import Path
from xml.etree import ElementTree as ET
from bz2 import BZ2File
import time
from nltk import sent_tokenize, word_tokenize
import wikitextparser as wtp
import regex as re
import numpy as np
import scipy as sp
import multiprocessing as mltp
from multiprocessing import Pool
import pickle

wikiroot = Path() / ".." / "wiki_data"
wikipaths = list(wikiroot.glob("**/wikilines-100000*.txt"))
wikifiles = [p.resolve() for p in wikipaths]

EMBEDDING_DIMENSION = 100

embroot = Path() / ".." / "downloads"
embpath = embroot / "glove.6B.{}d.txt".format(EMBEDDING_DIMENSION)

embedding = {}
for line in open(embpath.resolve(), 'r'):
    parts = line.split()
    embedding[parts[0]] = np.asarray(parts[1:]).astype('float32')
print("Loaded Embeddings.")
    
template_regex = "\{\{([^\|\}]*)\|?([^\|\}]*)\}\}"
wikilink_regex = "\[\[([^\|\]]*)\|?([^\|\]]*)\]\]"

def pretty_print(root, prefix):
    print("{}<{}>".format(prefix, root.tag))
    for child in root:
        pretty_print(child, "\t{}".format(prefix))
    if root.text != None:
        print("\t{}{}".format(prefix, root.text))
    print("{}</{}>".format(prefix, root.tag))
    
class WikiStats(object):
    def __init__(self):
        # Surface = String
        # Title = String
        # Category = String
        self.redirect_table = {} # Map String -> Title
        self.title_freq = {} # Map Title -> Int
        self.surface_freq = {} # Map Surface -> Int
        self.surface_title_freq = {} # Map Surface -> (Map Title -> Int)
        self.context_sums = {} # Map Title -> NP.Vec
        self.context_count = {} # Map Title -> Int
        self.topics = {} # Map Title -> [Category]
        self.titles = set([])
        self.surfaces = set([])
        self.__cleaned = False
        
    def get_average_context(self, title):
        return self.context_sums[title] / self.context_count[title]
    
    def get_title_freq(self, title):
        title = title.lower()
        res = 0
        if title in self.title_freq:
            res += self.title_freq[title]
        if title in self.redirect_table:
            redir_to = self.redirect_table[title]
            if redir_to in self.title_freq:
                res += self.title_freq[redir_to]
        return res
    
    def get_title_prob(self, title):
        return self.title_freq[title] / sum(self.title_freq.values())
    
    def get_surface_title_prob(self, surface, title):
        return self.surface_title_freq[surface][title] / sum(self.surface_title_freq[surface].values())
    
    def get_title_probs(self):
        res = {}
        titles = self.titles
        total_mass = sum(self.title_freq.values())
        for title in titles:
            mass = self.title_freq[title]
            target = self.redirect_table[title] if title in self.redirect_table else title
            
            if target not in res:
                res[target] = 0
            res[target] += mass / total_mass
        return res
    
    def get_surface_probs(self):
        res = {}
        surfaces = self.surfaces
        total_mass = sum(self.surface_freq.values())
        for surface in surfaces:
            mass = self.surface_freq[surface]            
            res[surface] = mass / total_mass
        return res
    
    def get_surface_title_probs(self):
        res = {}
        titles = self.titles
        surfaces = self.surfaces
        for surface in surfaces:
            if surface not in res:
                res[surface] = {}
            
            total_mass = sum(self.surface_title_freq[surface].values())
            for title in self.surface_title_freq[surface]:
                mass = self.surface_title_freq[surface][title]
                target = self.redirect_table[title] if title in self.redirect_table else title
                
                if target not in res[surface]:
                    res[surface][target] = 0
                res[surface][target] += mass / total_mass
        return res
                
    def get_avg_contexts(self):
        res = {}
        titles = self.titles
        for title in titles:
            avg_context = self.context_sums[title] / self.context_count[title]
            target = self.redirect_table[title] if title in self.redirect_table else title
            
            if target not in res:
                res[target] = avg_context
            else:
                res[target] += avg_context
        return res    
    
    def merge(self, other):
        self.redirect_table = {} # Map String -> Title
        self.title_freq = {} # Map Title -> Int
        self.surface_freq = {} # Map Surface -> Int
        self.surface_title_freq = {} # Map Surface -> (Map Title -> Int)
        self.context_sums = {} # Map Title -> NP.Vec
        self.context_count = {} # Map Title -> Int
        self.topics = {} # Map Title -> [Category]
        self.titles = set([])
        self.surfaces = set([])
        
        self.titles = self.titles.union(other.titles)
        self.surfaces = self.surfaces.union(other.surfaces)
        
        for source in other.redirect_table:
            if source in self.redirect_table:
                print("Redirect collision: {} -> {} | {}".format(source, other.redirect_table[source], self.redirect_table[source]))
            self.redirect_table[source] = other.redirect_table[source]
            
        for title in other.title_freq:
            if title in self.title_freq:
                self.title_freq[title] += other.title_freq[title]
            else:
                self.title_freq[title] = other.title_freq[title]
                
        for surface in other.surface_freq:
            if surface in self.surface_freq:
                self.surface_freq[surface] += other.surface_freq[surface]
            else:
                self.surface_freq[surface] = other.surface_freq[surface]
                
        for surface in other.surface_title_freq:
            for title in other.surface_title_freq[surface]:
                if surface in self.surface_title_freq:
                    if title in self.surface_title_freq[surface]:
                        self.surface_title_freq[surface][title] += other.surface_title_freq[surface][title]
                    else:
                        self.surface_title_freq[surface][title] = other.surface_title_freq[surface][title]
                else:
                    self.surface_title_freq[surface] = other.surface_title_freq[surface]
                    
        for title in other.context_sums:
            if title in self.context_sums:
                self.context_sums[title] += other.context_sums[title]
            else:
                self.context_sums[title] = other.context_sums[title]
                
        for title in other.context_count:
            if title in self.context_count:
                self.context_count[title] += other.context_count[title]
            else:
                self.context_count[title] = other.context_count[title]
                
        return self
def remove_spans(text, start, ends):
        depth = 0
        res = ""
        ignoring = 0
        for i in range(len(text)):
            if text[i:i+len(start)] == start:
                depth += 1 
            if depth == 0:
                if ignoring == 0:
                    res += text[i]
                else:
                    ignoring -= 1
            for end in ends:
                if text[i:i+len(end)] == end:
                    depth -= 1
                    ignoring += len(end) - 1
                    break
        return res

def to_surface(text, verbose=False):
    '''
    Takes a MediaWiki sentence and produces a simplified surface form.
    '''
    
    def repl_fn(m):
        if m.group(2) == '':
            return m.group(1)
        else:
            return m.group(2)
    
    if verbose: print(text)
    text = remove_spans(text, '{{', ['}};', '}}'])
    text = remove_spans(text, '<ref', ['</ref>', '/>'])
    text = remove_spans(text, '[[Image:', [']]'])
    text = remove_spans(text, '[[File:', [']]'])
    text = re.sub(wikilink_regex, repl_fn, text)
    if verbose: print(text)

    return text

def tokenize(sentence):
    '''
    Tokenizes a wikipedia style sentence. Takes unprocessed wikipedia text and transforms it to a lowercase tokenized form
    '''
    return [w.lower() for w in word_tokenize(sentence)]

def embed_sentence(tokens):
    '''
    Takes in a list of tokens, returns a sentence embedding
    @tokens a list of word tokens, all lowercase, all nice words.
    @return a numpy array of size EMBEDDING_DIMENSION containing a sentence embedding.
    '''
    token_vecs = np.array([embedding[tk] for tk in tokens if tk in embedding])
    if len(token_vecs) == 0:
        return np.zeros((EMBEDDING_DIMENSION,))
    return np.mean(token_vecs, axis=0)

def section_break(text):
    '''
    Takes an input of text and breaks it into a list of sections.
    '''
    sections = []
    buffer = ""
    minibuf = ""
    buffering = True
    for i in range(len(text)-1):
        c = text[i]
        if buffering and c == '=' and text[i+1] == '=':
            buffering = False
            
        if buffering:
            buffer += c
        else:
            minibuf += c
            
        if len(minibuf) >= 5:
            if minibuf[:3] == '===':
                if minibuf[-3:] == '===':
                    sections.append(buffer)
                    buffer = ''
                    minibuf = ''
                    buffering = True
            elif minibuf[:2] == '==' and minibuf[-2:] == '==':
                sections.append(buffer)
                buffer = ''
                minibuf = ''
                buffering = True
                
    sections.append(buffer + minibuf)
    return sections
    
def parse_wiki(text):
    '''
    Takes in a string of wiki markup from a single article.
    Returns list of link info tuples: (surface, title, sentence embedding).
    @text An unprocessed wiki markup string, from one article.
    @return (a,b,c)
    @return a List of sentences as a string. Unprocessed.
    @return a list of links, minus the categories, to lowercase
    @return c List of categories, to lowercase.
    '''
    sections = section_break(text)
    res = []
    for section in sections:
        sentences = sent_tokenize(section.strip())
        for sentence in sentences:
            sentence = sentence.strip()
            parsed = wtp.parse(sentence)
            links = parsed.wikilinks
            surface = to_surface(sentence)
            if len(surface) > 0:
                tokens = tokenize(surface)
                embedding = embed_sentence(tokens)
                for link in links:
                    if link.text != None and ('|' in link.text or '[[' in link.text):
                        pass
                    else:
                        res.append((link.text, link.title, embedding))
    return res 
    
def process_text(name, text, wikistats):
    link_data = parse_wiki(text)
    for (surface, title, embedding) in link_data:
        
        wikistats.titles.add(title)
        wikistats.surfaces.add(surface)
        
        if title not in wikistats.title_freq:
            wikistats.title_freq[title] = 0
        wikistats.title_freq[title] += 1
        
        if surface not in wikistats.surface_freq:
            wikistats.surface_freq[surface] = 0
        wikistats.surface_freq[surface] += 1
        
        if surface not in wikistats.surface_title_freq:
            wikistats.surface_title_freq[surface] = {}
        if title not in wikistats.surface_title_freq[surface]:
            wikistats.surface_title_freq[surface][title] = 0
        wikistats.surface_title_freq[surface][title] += 1
        
        if title not in wikistats.context_sums:
            wikistats.context_sums[title] = np.zeros((EMBEDDING_DIMENSION,))
            wikistats.context_count[title] = 0
        wikistats.context_sums[title] += embedding
        wikistats.context_count[title] += 1
    return wikistats
            
def process_xml(root, wikistats):
    '''
    Given a tree for an article, process it and update our tables of information.
    '''
    name = root.find('title').text.lower()
    
    if 'isambiguation' in name:
        return ('disambiguation', wikistats)
    
    redir = root.find('redirect')
    if redir != None:
        redir_to = redir.get('title').lower()
        if name != redir_to:
            wikistats.redirect_table[name] = redir_to
        return ('redirect', wikistats)
    
    recent_rev = root.find('revision')
    if recent_rev != None:
        content = recent_rev.find('text')
        if content != None:
            text = content.text.lower()
            wikistats = process_text(name, text, wikistats)
            return ('article', wikistats)
        
    return ('err', wikistats)

def process_wikilinefile(wikilinefile, verbose=False, cutoff=None):
#     redirect_table = {} # Map String String, the correct redirect for each entry.
#     title_freq = {} # Map String Int
#     surface_freq = {} # Map String Int
#     surface_title_freq = {} # frequency of a specific surface form linking to a specific title. Map (String, String) Int
#     avg_context = {} # Map Title (NP.Array, Int) || array is a sum, int is the total count contributing to the sum.
#     topics = {} # Map Title [Category]
    wikistats = WikiStats()
    
    total_pages = 0
    total_articles = 0
    total_redirects = 0
    total_disambiguation = 0
    total_others = 0
    for line in open(wikilinefile, 'r'):
        root = ET.fromstring(line)
        total_pages += 1
        if total_pages % 10000 == 0:
            print("{}: Pages: {}".format(wikilinefile, total_pages))
        if cutoff != None and total_pages >= cutoff:
            return {'total_page_count' : total_pages,
                    'article_count' : total_articles,
                    'redirect_count' : total_redirects,
                    'disambiguation_count' : total_disambiguation,
                    'other_count' : total_others,
                    'wikistats' : wikistats}
        
        ns = root.find('ns').text
        if ns != '0':
            continue
        
        page_type, wikistats = process_xml(root, wikistats)
        if page_type == 'article': total_articles += 1
        elif page_type == 'redirect': total_redirects += 1
        elif page_type == 'disambiguation': total_disambiguation += 1
        else: total_others += 1
            
    return {'total_page_count' : total_pages,
            'article_count' : total_articles,
            'redirect_count' : total_redirects,
            'disambiguation_count' : total_disambiguation,
            'other_count' : total_others,
            'wikistats' : wikistats}

def get_stats(filepath):
    print("Getting stats from: {}".format(filepath))
    stats = process_wikilinefile(filepath)
    print("Got stats from: {}".format(filepath))
    return stats

if __name__ == '__main__':
    
    mltp.cpu_count()
    pool = Pool(processes=16)
    
    print("Starting pool at t={:0.2f}.".format(time.time()))
    poolstart = time.time()
    stat_objects = pool.map(get_stats, wikifiles)
    print("Pool finished in {:02f}s".format(time.time() - poolstart))
    
    mergestart = time.time()
    master = WikiStats()
    for i, stat_obj in enumerate(stat_objects):
        master.merge(stat_obj['wikistats'])
    print("Merged in {:0.2f}s".format(time.time() - mergestart))
        
    title_probs = master.get_title_probs()
    surface_title_probs = master.get_surface_title_probs()
    surface_probs = master.get_surface_probs()
    avg_contexts = master.get_avg_contexts()
    
    obj = {'title' : title_probs, 'surface_title' : surface_title_probs, 'surface' : surface_probs, 'context' : avg_contexts}
    dump_path = wikiroot / 'stats' / 'monolith.pkl'
    with open(dump_path, 'wb+') as fh:
        pickle.dump(obj, fh)
    print("Saved to {}".format(dump_path))
   