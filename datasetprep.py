__author__ = 'distro'

datapath="data/"
rawtrainfile="train_1000.label"
trainfile="train.csv"
trecfile="trec9temp.csv"
traindata=[]
with open(datapath+rawtrainfile) as ob:
    for line in ob.readlines():
        label,sep,sentence=line.partition(" ")
        sentence=sentence.strip()
        parentcat,childcat=label.strip().split(":")
        if parentcat=="ABBR":
            traindata.append((sentence,"WHAT"))
        elif parentcat=="DESC":
            traindata.append((sentence,"UNKNOWN"))
        elif parentcat=="ENTY":
            traindata.append((sentence,"WHAT"))
        elif parentcat=="HUM":
            traindata.append((sentence,"WHO"))
        elif parentcat=="LOC":
            traindata.append((sentence,"UNKNOWN"))
        elif parentcat=="NUM":
            if childcat=="date":
                traindata.append((sentence,"WHEN"))
            else:
                traindata.append((sentence,"UNKNOWN"))

with open(datapath+trecfile) as ob:
    for line in ob.readlines():
        datalist=line.strip().split("\t")
        if len(datalist)==2:
            sentence,label=datalist
            traindata.append((sentence.strip(),label.strip()))
        else:
            print "Ignoring"

with open(datapath+trainfile,"wb") as ob:
    for sentence,label in traindata:
        ob.write(sentence+"\t"+label+"\n")