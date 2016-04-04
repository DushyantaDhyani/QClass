# QClass

A python package to predict the Class of a Question i.e. whether **Who** , **What** , **When** , **Affirmation** or **Unknown**.

## Installation

Required packages (for Python 2.7.6)

```python
joblib==0.9.4
nltk==3.2
numpy==1.10.4
pandas==0.17.0
scikit-learn==0.17
scipy==0.17.0
sklearn==0.0
```

## Usage

### From Command Line

`python QClass.py "What is your name?"`

#### Output

`WHAT`

### From Inside Python Code

```python
from QClass import getQClass
queslist=[
    'Define metamorphosis',
    'What was the date of the match?',
    'Did John enjoy the match?',
    'Who invented electricity?',
    "Did you hear John's decided to go to business school?",
    'Explain centrifugal force',
    'What is the name of the god of water?',
    'Is it dry outside?',
]
for ques in queslist:
    print ques+"\t"+getQClass(ques)    
```

#### Output

```
Define metamorphosis	UNKNOWN
What was the date of the match ?	WHEN
Did John enjoy the match ?	Affirmation
Who invented electricity ?	WHO
Did you hear John's decided to go to business school ?	Affirmation
Explain centrifugal force	UNKNOWN
What is the name of the god of water ?	WHAT
Is it dry outside ?	Affirmation
```

## Training Your Own Classifier

Place your `train.csv` in the `data` folder in the following tab-separated format

`Question    Label`

Note: The Affirmation cases do not require training (they are heuristics based) and so should be included in the training file .

Run `python training.py` and the model would be saved in the `model` directory.

