import re
import string
import sys
import unicodedata
import json
from HTMLParser import HTMLParser
from bs4 import BeautifulSoup


def load_json(filename):
    with open(filename, 'r') as fin:
        db = json.load(fin)
    return db


def save_json(filename, data):
    with open(filename, 'w') as fout:
        json.dump(data, fout)


def build_table():
    table = dict.fromkeys(
        i for i in xrange(sys.maxunicode) if not unicodedata.category(
            unichr(i)).startswith('L'))

    for id in table:
        table[id] = unicode(' ')
    return table

# Remove all punctuation from text
def remove_punctuation(text):
    text = text.translate(table)
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = re.sub(' +', ' ', text)
    return text


def test_trans(s):
    table = string.maketrans("", "")
    return s.translate(table, deletechars=string.punctuation)


def get_qa_pair(result_db):
    f_result_db = dict()
    for p_id in result_db:
        f_result_db[p_id] = dict()
        f_result_db[p_id]['question'] = remove_punctuation(
            result_db[p_id]['subject'])
        f_result_db[p_id]['answer'] = remove_punctuation(
            result_db[p_id]['answer'])
	print len(f_result_db)
	return f_result_db



# Convert json file into IBM Model readable format
def build_model_input(db,fout_name):
    with open(fout_name, 'w') as fout:
        cnt = 0
        for post_id in db:
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


# Build word2vec input from clean json file
def json2wordvec():
    db = load_json('result-1.json')

    with open('yahoo-text', 'w') as fout:
        for id in db:
            line = (db[id]['answer'] + db[id]['question']) + ' '
            fout.write(line.lower().encode('utf8'))


# Read from original Webscope_L4 format file
# Output
def parse_text(res, obj, pair_id, key):
    try:
        res[pair_id][key] = h.unescape(obj.get_text())
    except:
        res[pair_id][key] = None


# from raw input to json
# remove all html tag
def raw2json():
    for i in range(2, 5):
        input_filename = 'manner-small-1-{0}.xml'.format(i)
        bs_obj = BeautifulSoup(open(input_filename), 'html5lib')
        print(len(bs_obj.find_all('vespaadd')))
        result = dict()
        item_list = bs_obj.find_all('vespaadd')
        for item in item_list:
            pair_id = item.uri.get_text()
            result[pair_id] = dict()
            parse_text(result, item.subject, pair_id, 'subject')
            parse_text(result, item.content, pair_id, 'content')
            parse_text(result, item.maincat, pair_id, 'maincat')
            parse_text(result, item.cat, pair_id, 'cat')
            parse_text(result, item.subcat, pair_id, 'subcat')
            parse_text(result, item.bestanswer, pair_id, 'answer')

        with open('{0}.json'.format(i), 'w')as fout:
            json.dump(result, fout)


def split_category(db):
    cat_db = dict()
    for key in db:
        cat = db[key]['maincat']
        if cat in cat_db:
            cat_db[cat].append(db[key])
        else:
            cat_db[cat] = [db[key]]
    return cat_db


def grasp_category(cat_db, name):
    bus_db = dict()
    count = 0
    for line in cat_db[name]:
        bus_db[str(count)] = line
        count += 1
    return bus_db


def get_certain_cat(cat_name='Business & Finance'):
    db = load_json('result.json')
    bus_cat = grasp_category(split_category(db), cat_name)
    save_json('{0}_db.json'.format(cat_name), bus_cat)

table = build_table()
input = load_json('bus_db.json')
qa_pair = get_qa_pair(input)
build_model_input(qa_pair,'YahooBusPost.dat')


