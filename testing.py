__author__ = 'distro'
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
    'How did you arrive here?',
    'Did you invite Mary?',
    'When is the show happening?',
    'Do you not want a cigar?',
    'Is there a cab available for the airport?',
    'Name the god of water',
    'What is your name?',
    'You guys must be starving. You want to get something to eat?',
    "I'd like to take you guys out to dinner while I'm here- we'd have time to go somewhere around here before the evening session tonight,don't you think?",
 ]
for ques in queslist:
    print ques+"\t"+getQClass(ques)