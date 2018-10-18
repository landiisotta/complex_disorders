import os
import csv
import pickle
import re 

data_folder = os.path.expanduser('~/data1/complex_disorders/autism')
file_names = ['noteID_autism_isotta.csv', 'note_term_sequence.csv', 'str2cui_isotta.csv']

with open(os.path.join(data_folder, file_names[0])) as f:
    rd = csv.reader(f)
    mrn_notes = {}
    for r in rd:
        mrn_notes.setdefault(r[1], list()).append(r[0])

with open(os.path.join(data_folder, file_names[1])) as f:
    rd = csv.reader(f)
    next(rd)
    note_term = {}
    for r in rd:
        note_term.setdefault(r[0], list()).append(r[1])

with open(os.path.join(data_folder, file_names[2]), encoding='utf-8') as f:
    rd = csv.reader(f, delimiter='@', 
                    quotechar='"', 
                    quoting=csv.QUOTE_ALL, 
                    skipinitialspace=True,
                    escapechar='\\')
    str2cui = {}
    for r in rd:
        str2cui.setdefault(r[0], set()).add(r[1])

mrn_terms = {}
not_found = []
for mrn in mrn_notes:
    for n in mrn_notes[mrn]:
        try:
            mrn_terms.setdefault(mrn, list()).append(note_term[n][0])
        except KeyError:
            not_found.append(n)
            pass

mrn_notesText = {}
for mrn in mrn_terms:
    for seq in mrn_terms[mrn]:
        l = []
        try:
            for t in str.split(seq, '|'):
                if t.find('n') != -1:
                    l.append('NOT ' + list(str2cui[t.split('n')[1]])[0])
                elif t.find('f') != -1:
                    l.append('Family history ' + list(str2cui[t.split('f')[1]])[0])
                else:
                    l.append(list(str2cui[t])[0])
        except KeyError:
            pass
        mrn_notesText.setdefault(mrn, list()).append(' '.join(l))

# with open(os.path.join(data_folder, 'tmp_file-mrn_notesText'), 'wb') as f:
#     pickle.dump(mrn_notesText, f)

# ####TRY THIS ONE
# with open(os.path.join(data_folder, 'tmp_file-mrn_notesText'), 'rb') as f:
#     mrn_notesText = pickle.load(f)

with open(os.path.join(data_folder, 'clinical-notes_asd.csv'), 'w') as f:
    wr = csv.writer(f, delimiter='\t')
    for mrn in mrn_notesText:
        for text in mrn_notesText[mrn]:
            tmp = re.sub('[^A-Za-z0-9]+', ' ', text) 
            wr.writerow([mrn, tmp])
                
