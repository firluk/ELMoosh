#################################################################
#            IR project Spring 2019 - evaluation script         #
#################################################################

import json
import sys

def bsearch(sequence, value):
    lo, hi = 0, len(sequence) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sequence[mid]['id'] < value:
            lo = mid + 1
        elif value < sequence[mid]['id']:
            hi = mid - 1
        else:
            return mid
    return None

for arg in sys.argv:
	if arg == "-h":
		print("Usage: eval.py <questions filepath> <answers filepath>")
		sys.exit()

questions = [line.rstrip('\n') for line in open(sys.argv[1],'r', encoding='utf-8')]
answers = sorted(json.load(open(sys.argv[2],'r', encoding='utf-8')), key=lambda k: k['id'])

sumAccuracy = 0.0;
sumMRR = 0.0;

first_time_ignore = True
for q in questions:
    #ignore first metadata line
    if first_time_ignore:
        first_time_ignore = False
        continue
    qrl = q.split('\t');
    doc_id = qrl[2]
    relanswers = set()
    for psg_id in qrl[4].split(','):
        relanswers.add(doc_id + ':' + psg_id)
    # eval questions
    inx = bsearch(answers, qrl[0])
    if inx is None:
        print("No answers for question id: " + qrl[0] + ". This counts as no match!")
        continue
    qanswers = sorted(answers[inx]['answers'], key=lambda k: k['score'], reverse=True)
    rank = 1
    matchFound = False
    for asw in qanswers:
        if asw['answer'] in relanswers:
            matchFound = True
            break
        rank += 1
        if rank == 6:
            break
    if matchFound:
        if rank == 1:
            sumAccuracy += 1
        sumMRR += 1.0 / rank

num_questions = len(questions) - 1
meanAcc = sumAccuracy / num_questions;
meanMRR = sumMRR / num_questions;
print("Accuracy: {}".format(meanAcc))
print("MRR@5: {}".format(meanMRR))