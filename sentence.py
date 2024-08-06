import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
class Sentence:
    def __init__(self, sentence_id, phrase, sentiment_class,num_classes):
        self.sentence_id = sentence_id
        self.phrase = self.create_phrase_dictionary(phrase)
        self.sentiment_class = self.map_sentiment_class(sentiment_class,num_classes)

    def map_sentiment_class(self,sentiment_class,num_classes): # used as Sentence.map_sentiment_class
        """
        Maps sentiment class to correct value if number of classes is 3

        Parameters :
        sentiment_class : Integer
                    Current sentiment class value assigned to Sentence
        num_classes : Integer
                    Number of classes specified for use in the classifier
        """
        if num_classes == 3:
            if sentiment_class <= 1:
                return 0
            elif sentiment_class == 2:
                return 1
            else:
                return 2
        else:
            return sentiment_class

    def create_phrase_dictionary(self,phrase):
        """
        Create dictionary out of the given sentence, with a word being mapped to its count
        Includes stemming and stop word removal

        Parameters :
        ----------
        phrase : string
                String to be converted to a dictionary of term counts
        """
        stemmer = SnowballStemmer("english")
        phrase_dict = dict()
        stop_words = set(stopwords.words('english'))
        phrase = phrase.split(" ")
        for word in phrase:
            if (word not in stop_words and word != ""):
                word = stemmer.stem(word)
                if word in phrase_dict:
                    phrase_dict[word] = phrase_dict[word] + 1
                else:
                    phrase_dict[word] = 1
        return phrase_dict
        
