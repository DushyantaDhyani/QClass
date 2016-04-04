__author__ = 'distro'
import sys
import re
import pandas as pd
import nltk

MODEL_PATH="model/"
MODEL_FILE="classifier.pkl"
TFIDF_FILE="tfidf.pkl"
QUESLE_FILE="quesle.pkl"
ValidQuesWords=['A', 'About', 'Can', 'Define', 'For', 'Give', 'How', 'In', 'Name', 'On', 'Tell', 'The', 'To', 'What', 'Whats', 'When', 'Where', 'Wheres', 'Which', 'Who', 'Whom', 'Whos', 'Whose', 'Why']
ClassMapper={"UNKNOWN":0,"WHAT":1,"WHEN":2,"WHO":3}

def getInverseClassMapper(Mapper):
    RevMap={}
    for key in Mapper:
        RevMap[Mapper[key]]=key
    return RevMap

def clean(ques):
    ques=ques.rstrip('?:!.,;')
    ques=re.sub('[!@#$,\`\']', '', ques)
    return ques.lower().strip()

def loadModel(path):
    from sklearn.externals import joblib
    return joblib.load(path)

def getQuesWord(ques):
    word=ques.strip().split(" ")[0].strip()
    if word in ValidQuesWords:
        return word
    else:
        return "InvalidQues"

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def wordCount(ques):
    return len(re.findall(r'\w+', ques))

def getPosFeatures(ques):
    featurelist=[]
    poslist=[]
    for token,pos in nltk.pos_tag(nltk.word_tokenize(ques)):
        poslist.append(pos)

    # Counting Adjectives, having POS Tags JJ/JJS/JJR
    count=0
    for pos in poslist:
        if pos in ["JJ","JJS","JJR"]:
            count+=1
    featurelist.append(count)

    # Counting Nouns, having POS Tags NN/NNS/NNP/NNPS
    count=0
    for pos in poslist:
        if pos in ["NN","NNS","NNP","NNPS"]:
            count+=1
    featurelist.append(count)

    # Counting Personal pronoun, Having POS Tags PRP
    count=0
    for pos in poslist:
        if pos=="PRP":
            count+=1
    featurelist.append(count)

    # Counting Possessive pronoun, Having POS Tags PRP$
    count=0
    for pos in poslist:
        if pos=="PRP$":
            count+=1
    featurelist.append(count)

    # Counting Adverbs, Having POS Tags RB/RBS/RBR/RP
    count=0
    for pos in poslist:
        if pos in ["RB","RBS","RBR","RP"]:
            count+=1
    featurelist.append(count)

    # Counting Verb, base form, Having POS Tags VB
    count=0
    for pos in poslist:
        if pos=="VB":
            count+=1
    featurelist.append(count)


    # Counting Verb, past tense, Having POS Tags VBD
    count=0
    for pos in poslist:
        if pos=="VBD":
            count+=1
    featurelist.append(count)


    # Counting Verb, gerund or present participle, Having POS Tags VBG
    count=0
    for pos in poslist:
        if pos=="VBG":
            count+=1
    featurelist.append(count)


    # Counting Verb, past participle, Having POS Tags VBN
    count=0
    for pos in poslist:
        if pos=="VBN":
            count+=1
    featurelist.append(count)


    # Counting Verb, non-3rd person singular present, Having POS Tags VBP
    count=0
    for pos in poslist:
        if pos=="VBP":
            count+=1
    featurelist.append(count)


    # Counting Verb, 3rd person singular present, Having POS Tags VBZ
    count=0
    for pos in poslist:
        if pos=="VBZ":
            count+=1
    featurelist.append(count)

    # Counting Verbs, Having POS Tags VB/VBD/VBG/VBN/VBP/VBZ

    count=0
    for pos in poslist:
        if pos in ["VB","VBD","VBG","VBN","VBP","VBZ"]:
            count+=1
    featurelist.append(count)


    # Counting Wh-determiner, Having POS Tags WDT
    count=0
    for pos in poslist:
        if pos=="WDT":
            count+=1
    featurelist.append(count)

    # Counting Wh-pronoun, Having POS Tags WP
    count=0
    for pos in poslist:
        if pos=="WP":
            count+=1
    featurelist.append(count)


    # Counting Wh-adverb, Having POS Tags WRB
    count=0
    for pos in poslist:
        if pos=="WRB":
            count+=1
    featurelist.append(count)


    # Counting Wh-Tokens, Having POS Tags WDT/WP/WRB
    count=0
    for pos in poslist:
        if pos in ["WDT","WP","WRB"]:
            count+=1
    featurelist.append(count)
    return featurelist

def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'node') and t.node:
        if t.node in ['GPE','PEOPLE','ORGANIZATION']:
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names

def getNERFeatures(ques):
    namedEnt = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(ques)))
    return len(extract_entity_names(namedEnt))

def checkAffirmation(ques):
    pattern=r'[a-z]* (anyone |anybody |you)[a-z]*(tell|know)[a-z]*'
    BeVerbs=["is","am","are","was","were","been","being"]
    ModalVerbs=["can","could","shall","should","will","would","may","might"]
    AuxVerbs=["do","did","does","have","had","has"]
    if ques.startswith(tuple(BeVerbs)) or ques.startswith(tuple(ModalVerbs)) or ques.startswith(tuple(AuxVerbs)):
        if ques.startswith(tuple(BeVerbs)) and " or " in ques:
            return False
        else:
            if re.search(pattern,ques):
                return False
            else:
                return True
    else:
        return False

def getFeatures(ques):
    tvf=loadModel(MODEL_PATH+TFIDF_FILE)
    quesle=loadModel(MODEL_PATH+QUESLE_FILE)
    features=tvf.transform([ques]).toarray().flatten().tolist()
    features.append(quesle.transform(getQuesWord(ques)))
    features.append(hasNumbers(ques))
    features.append(wordCount(ques))
    features.extend(getPosFeatures(ques))
    features.append(getNERFeatures(ques))
    return pd.DataFrame([features])

def getWhClass(ques):
    clf=loadModel(MODEL_PATH+MODEL_FILE)
    df=getFeatures(ques)
    InverseClassMapper=getInverseClassMapper(ClassMapper)
    return InverseClassMapper[clf.predict(df)[0]]

def getQClass(ques):
    ques=clean(ques)
    if checkAffirmation(ques):
        return "Affirmation"
    else:
        return getWhClass(ques)

if __name__=="__main__":
    if len(sys.argv)==1:
        print "Please pass an input string"
    elif len(sys.argv)>2:
        print "Incorrect Format!! Enclose input string in double quotes"
    else:
        ques=sys.argv[1]
        print getQClass(ques)

