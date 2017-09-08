import json

prob_table = list()
with open('117-09-06.141405.tang.actual.ti.final', 'r') as fin:
    for line in fin:
        line = line.rstrip()
        line = line.lstrip()
        line = line.split(' ')
        if 'NULL' not in line and float(line[2]) > 0.001:
            prob_table.append(line)

with open('prob_table.json', 'w') as fout:
    json.dump(prob_table, fout)
