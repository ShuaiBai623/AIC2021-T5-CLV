import json
import sys
with open(sys.argv[1]) as f:
    train = json.load(f)
import spacy
nlp = spacy.load("en_core_web_sm")

track_ids = list(train.keys())
for id_ in track_ids:
	new_text = ""
	for i,text in enumerate(train[id_]["nl"]):
		doc = nlp(text)

		for chunk in doc.noun_chunks:
			nb = chunk.text
			break
		train[id_]["nl"][i] = nb+'. '+train[id_]["nl"][i]
		new_text += nb+'.'
		if i<2:
			new_text+=' '
	train[id_]["nl"].append(new_text)
		


with open(sys.argv[1].split('.')[-2]+"_nlpaug.json", "w") as f:
    json.dump(train, f,indent=4)
