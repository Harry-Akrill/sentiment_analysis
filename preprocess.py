import re
import nltk
# from nltk.corpus import stopwords
# from nltk.corpus.reader import WordListCorpusReader
# from nltk.corpus.reader.api import *
# from nltk.corpus import opinion_lexicon
class Preprocess:
    def __init__(self,phrase,features,training):
        self.features = features
        self.training = training
        self.processed_phrase = self.process_phrase(phrase)
    
    def punctuation_regex(self,text):
        """
        Remove punctuation from given text

        Parameters :
        ----------
        text : string
            String to have punctuation removed from
        """
        characters = r"[-,(.;:)@\#?\\|+_*`=~<>/&$]+" # doesnt remove apostraphes or exclamation marks
        return re.sub(characters, '', text)

    def lowercase_regex(self,text):
        """
        Set a string to lower case

        Parameters :
        ----------
        text : string
            String to be set to lower case
        """
        return re.sub(r"[A-Z]", lambda x: x.group(0).lower(), text)

    def decontracted(self,text):
        """
        Function implemented by Yann Dubois (Accessed 07/12/2022)
        https://stackoverflow.com/questions/43018030/replace-apostrophe-short-words-in-python

        Decontracts words within a sentences

        Parameters :
        ----------
        text : string
            String to be decontracted
        """
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"can\'t", "can not", text)

        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        return text

    def process_phrase(self,text):
        """
        Process phrase using pre-processing functions

        Parameters :
        ----------
        text: string
            String to be pre-processed
        """
        phrase = self.decontracted(self.punctuation_regex(self.lowercase_regex(text)))

        if self.training == False: # if you arent doing the special features stuff, just general processing
            return phrase

        # feature selection
        word_type_list = ["JJS","JJR","JJ","NN","RB","RBS","RBR","VB","VBD","NNS","VBG"]
        features = self.features 
        
        if features == "features":
            # positive_words = opinion_lexicon.positive() # Positive / Negative lexicon, Failed implementation, too computationally expensive
            # negative_words = opinion_lexicon.negative()
            split_phrase = phrase.split(" ")
            pos_tagged = nltk.pos_tag(split_phrase) # Tags a sentences with its "word types" e.g. adjectives
            new_phrase = ""
            for x,y in pos_tagged:
                if y in word_type_list:
                    if len(new_phrase) == 0:
                        new_phrase = x
                    else:
                        new_phrase = new_phrase + " " + x
            phrase = new_phrase
        return phrase
    