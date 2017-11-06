# Runs on python2
import json
import re
import sys
import unicodedata
from HTMLParser import HTMLParser
from bs4 import BeautifulSoup


def load_json(filename):
    with open(filename, 'r') as fin:
        db = json.load(fin)
    return db


def save_json(obj, filename):
    with open(filename, 'w') as fout:
        json.dump(obj, fout)


def build_table():
    table = dict.fromkeys(
        i for i in xrange(sys.maxunicode) if not unicodedata.category(
            unichr(i)).startswith('L'))

    for id in table:
        table[id] = unicode(' ')
    return table


TABLE = build_table()
h = HTMLParser()


# Remove all punctuation from text
def remove_punctuation(text):
    if text is not None:
        text = text.translate(TABLE)
        text = text.replace('\n', '')
        text = text.replace('\r', ' ')
        text = re.sub(' +', ' ', text)
    return text



# Clean html tags
def parse_text(pair, obj, key):
    h = HTMLParser()
    try:
        pair[key] = h.unescape(obj.get_text())
    except:
        pair[key] = None


def get_qa_pair(db):
    result = dict()
    for p_id in db:
        if db[p_id]['subject'] is None or db[p_id]['answer'] is None:
            continue
        result[p_id] = dict()
        result[p_id]['question'] = remove_punctuation(
            db[p_id]['subject'])
        result[p_id]['answer'] = remove_punctuation(
            db[p_id]['answer'])
    print len(result)
    return result


# Convert Q&A pair into IBM model input
def IBM_input(db, fout_name):
    with open(fout_name, 'w') as fout:
        cnt = 0
        for post_id in db:
            if(len(db[post_id]['answer'])>5 and len(db[post_id]['question'])>5):
                line = {"content": db[post_id]['answer'],
                        "source": "1",
                        "title": db[post_id]['question'],
                        "date": "1",
                        "timestamp": "1",
                        "resourceKey": "1",
                        "tags": ['1'],
                        "id": str(cnt),
                        "summary": "1"}
                fout.writelines(json.dumps(line) + '\n')
                cnt = cnt + 1


# Convert Q&A pair into word2vec model input
def W2C_input(db, out_filename):
    with open(out_filename, 'w') as fout:
        for id in db:
            line = (db[id]['answer'] + db[id]['question']) + ' '
            fout.write(line.lower().encode('utf8'))


# Convert xml file into relative database
def raw2db(input_filename):
    bs_obj = BeautifulSoup(open(input_filename), 'html5lib')
    result = dict()
    item_list = bs_obj.find_all('vespaadd')
    length = len(item_list)
    count = 0
    print length, ' in total:'
    for item in item_list:
        pair_id = item.uri.get_text()
        result[pair_id] = dict()
        parse_text(result[pair_id], item.subject, 'subject')
        parse_text(result[pair_id], item.content, 'content')
        parse_text(result[pair_id], item.maincat, 'maincat')
        parse_text(result[pair_id], item.cat, 'cat')
        parse_text(result[pair_id], item.subcat, 'subcat')
        parse_text(result[pair_id], item.bestanswer, 'answer')

        count += 1
        if count % 1000 == 0:
            print '{0} of {1} processed'.format(count, length)
    return result


# Convert dicted xml into relative database
def parse_text2(pair, obj, key):
    if obj == '':
        pair[key]=None
    else:
        pair[key] = h.unescape(obj)

def dict2db(db):
    result = dict()
    item_list = db['data']['vespaadd']
    length = len(item_list)
    count = 0
    print length, ' in total:'
    for item in item_list:
        try:
            item = item['document']
            pair_id = item['uri']
            result[pair_id] = dict()
            parse_text2(result[pair_id], item.get('subject', ''), 'subject')
            parse_text2(result[pair_id], item.get('content', ''), 'content')
            parse_text2(result[pair_id], item.get('maincat', ''), 'maincat')
            parse_text2(result[pair_id], item.get('cat', ''), 'cat')
            parse_text2(result[pair_id], item.get('subcat', ''), 'subcat')
            parse_text2(result[pair_id], item.get('bestanswer', ''), 'answer')
        except KeyError:
            print item
            break

        count += 1
        if count % 1000 == 0:
            print '{0} of {1} processed'.format(count, length)
    return result


# Arrange pair in category
def split_category(db):
    cat_db = dict()
    for key in db:
        cat_name = db[key]['maincat']
        if cat_name in cat_db:
            cat_db[cat_name][key] = db[key]
        else:
            cat_db[cat_name] = dict()
            cat_db[cat_name][key] = db[key]
    return cat_db

def get_much_input():
    bus_db = dict()
    count = 0
    for i in range(30):
        db = load_json('work/L6-db/{0}-db.json'.format(i))
        for k in db:
            if db[k]['maincat'] == 'Business & Finance':
                bus_db[k] = db[k]
                count += 1
                if count % 10000 == 0:
                    print count, ' items processed'
