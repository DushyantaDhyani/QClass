__author__ = 'distro'
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import nltk

def clean(ques):
    ques=ques.rstrip('?:!.,;')
    ques=re.sub('[!@#$,\`\']', '', ques)
    return ques.lower().strip()

def getInverseClassMapper(Mapper):
    RevMap={}
    for key in Mapper:
        RevMap[Mapper[key]]=key
    return RevMap

def getQuesWord(ques):
    word=ques.strip().split(" ")[0].strip()
    if word in ValidQuesWords:
        return word
    else:
        return "InvalidQues"

def getLastWord(ques):
    return ques.strip().split(" ")[-1].strip()

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def wordCount(ques):
    return len(re.findall(r'\w+', ques))

def getPosFeatures(queslist):
    posfeaturelist=[]
    for ques in queslist:
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


        posfeaturelist.append(featurelist)

    return pd.DataFrame(posfeaturelist,columns=PosFeatureNames)

def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'node') and t.node:
        if t.node in ['GPE','PEOPLE','ORGANIZATION']:
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names

def getNERFeatures(queslist):
    nerfeaturelist=[]
    for ques in queslist:
        namedEnt = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(ques)))
        nerfeaturelist.append(len(extract_entity_names(namedEnt)))
        # break
    return nerfeaturelist

def saveModel(model,path):
    from sklearn.externals import joblib
    joblib.dump(model, path)

datapath="data/"
modelpath="model/"
modelfile="classifier.pkl"
trainfile="train.csv"
tfidffile="tfidf.pkl"
queslefile="quesle.pkl"
ClassMapper={"UNKNOWN":0,"WHAT":1,"WHEN":2,"WHO":3}
svmparams = {'kernel':('linear', 'rbf'), 'C':[0.1,1, 10], 'class_weight':[None,'balanced']}
PosFeatureNames=["AdjectiveCount","NounCount","PRP","PRPDollar","Adverb","VB","VBD","VBG","VBN","VBP","VBZ","VERB","WDT","WP","WRB","WCOMPONENT"]
StaticFeatureList=['HasNumbers', 'WordCount', 'AdjectiveCount', 'NounCount', 'PRP', 'PRPDollar', 'Adverb', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VERB', 'WDT', 'WP', 'WRB', 'WCOMPONENT', 'NamedEntityCount']
ValidQuesWords=['A', 'About', 'Can', 'Define', 'For', 'Give', 'How', 'In', 'Name', 'On', 'Tell', 'The', 'To', 'What', 'Whats', 'When', 'Where', 'Wheres', 'Which', 'Who', 'Whom', 'Whos', 'Whose', 'Why']
InverseClassMapper=getInverseClassMapper(ClassMapper)
print "Reading and Cleaning Data"
df=pd.read_csv(datapath+trainfile,sep="\t",header=None,names=["question","class"])
df["question"]=df["question"].apply(lambda x:clean(x))
df["class"]=df["class"].apply(lambda x:ClassMapper[x.strip()])
# getStats(list(df["question"]),list(df["class"]))
print "Generating general Features"
df["quesword"]=df["question"].apply(lambda x:getQuesWord(x))
df["lastword"]=df["question"].apply(lambda x:getLastWord(x))
df["HasNumbers"]=df["question"].apply(lambda x:hasNumbers(x))
df["WordCount"]=df["question"].apply(lambda x:wordCount(x))
print "Generating POS Features"
df=df.join(getPosFeatures(list(df["question"])))
print "Generating NER Features"
df["NamedEntityCount"]=getNERFeatures(list(df["question"]))
print "Generating TF-IDF Features"
vocabulary=list(df["question"])
tfv = TfidfVectorizer(
        min_df=2,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 2),
        use_idf=True,
        smooth_idf=1,
        sublinear_tf=1,
        lowercase=False
        # stop_words = 'english'
)
tfv.fit(vocabulary)
X =  tfv.transform(vocabulary).toarray()
saveModel(tfv,modelpath+tfidffile)
X=pd.DataFrame(X)
print "Encoding Certain Features"
# lastwordle=LabelEncoder()
# lastwordle.fit(df["lastword"])
# X["lastword"]=lastwordle.transform(df["lastword"])
queswordle=LabelEncoder()
queswordle.fit(df["quesword"])
saveModel(queswordle,modelpath+queslefile)
X["quesword"]=queswordle.transform(df["quesword"])
X=X.join(df[StaticFeatureList])
Y=df["class"]
print X.shape
print "Training Model"
svc = svm.SVC()
clf=svm.SVC(kernel='linear',verbose=True)
clf.fit(X,Y)
print "Saving Model"
saveModel(clf, modelpath+modelfile)