from __future__ import division
from collections import Counter
class SentenceSet:
    def __init__(self, sentences,num_classes,features):
        self.sentences = sentences
        self.num_classes = num_classes
        self.features = features
        self.f_likelihoods = self.feature_likelihoods()

    def feature_likelihoods(self):
        """
        Calculates feature likelihoods of the words 
        in the set of sentences within the SentenceSet object
        """
        num_classes = self.num_classes
        sentences = self.sentences
        if num_classes == 3:
            positive_dict = dict()
            neutral_dict = dict()
            negative_dict = dict()

            #getting count of each class dictionary
            for sentence in sentences:
                phrase = sentence.phrase
                if sentence.sentiment_class == 0:
                    negative_dict = Counter(negative_dict) + Counter(phrase) # total negative sentiment words
                if sentence.sentiment_class == 1:
                    neutral_dict = Counter(neutral_dict) + Counter(phrase)
                if sentence.sentiment_class == 2:
                    positive_dict = Counter(positive_dict) + Counter(phrase)

            num_positive_values = sum(positive_dict.values())
            num_neutral_values = sum(neutral_dict.values())
            num_negative_values = sum(negative_dict.values())
            num_distinct_total_values = len(Counter(negative_dict) + Counter(neutral_dict) + Counter(positive_dict))

            #calculating likelihoods by dividing by total counts + class dictionary counts
            for word in positive_dict:
                positive_dict[word] = (positive_dict[word] + 1) / (num_positive_values + num_distinct_total_values) # Laplace smoothing
            for word in neutral_dict:
                neutral_dict[word] = (neutral_dict[word] + 1) / (num_neutral_values + num_distinct_total_values)
            for word in negative_dict:
                negative_dict[word] = (negative_dict[word] + 1) / (num_negative_values + num_distinct_total_values)
            
            all_words = Counter(positive_dict) + Counter(neutral_dict) + Counter(negative_dict)
            
            for word in all_words: # add in all words not included in class dictionary but included in total words between 3 dictionaries
                if word not in positive_dict:
                    positive_dict[word] = 1 / (num_positive_values + num_distinct_total_values)
                if word not in neutral_dict:
                    neutral_dict[word] = 1 / (num_neutral_values + num_distinct_total_values)
                if word not in negative_dict:
                    negative_dict[word] = 1 / (num_negative_values + num_distinct_total_values)
            return [negative_dict,neutral_dict,positive_dict]
        
        if num_classes == 5:
            positive_dict = dict()
            fairly_positive_dict = dict()
            neutral_dict = dict()
            fairly_negative_dict = dict()
            negative_dict = dict()

            for sentence in sentences:
                phrase = sentence.phrase
                if sentence.sentiment_class == 0:
                    negative_dict = Counter(negative_dict) + Counter(phrase)
                if sentence.sentiment_class == 1:
                    fairly_negative_dict = Counter(fairly_negative_dict) + Counter(phrase)
                if sentence.sentiment_class == 2:
                    neutral_dict = Counter(neutral_dict) + Counter(phrase)
                if sentence.sentiment_class == 3:
                    fairly_positive_dict = Counter(fairly_positive_dict) + Counter(phrase)
                if sentence.sentiment_class == 4:
                    positive_dict = Counter (positive_dict) + Counter(phrase)

            num_positive_values = sum(positive_dict.values())
            num_fairly_positive_values = sum(fairly_positive_dict.values())
            num_neutral_values = sum(neutral_dict.values())
            num_fairly_negative_values = sum(fairly_negative_dict.values())
            num_negative_values = sum(negative_dict.values())
            num_distinct_total_values = len(Counter(negative_dict) + Counter(neutral_dict) + Counter(positive_dict)
                                            +Counter(fairly_negative_dict) + Counter(fairly_positive_dict))
                    
            for word in positive_dict:
                positive_dict[word] = (positive_dict[word] + 1) / (num_positive_values + num_distinct_total_values)
            for word in fairly_positive_dict:
                fairly_positive_dict[word] = (fairly_positive_dict[word] + 1) / (num_fairly_positive_values + num_distinct_total_values) # Laplace smoothing
            for word in neutral_dict:
                neutral_dict[word] = (neutral_dict[word] + 1) / (num_neutral_values + num_distinct_total_values)
            for word in fairly_negative_dict:
                fairly_negative_dict[word] = (fairly_negative_dict[word] + 1) / (num_fairly_negative_values + num_distinct_total_values)
            for word in negative_dict:
                negative_dict[word] = (negative_dict[word] + 1) / (num_negative_values + num_distinct_total_values)
                
            all_words = Counter(positive_dict) + Counter(neutral_dict) + Counter(negative_dict) + Counter(fairly_positive_dict) + Counter(fairly_negative_dict)
            for word in all_words: # add in all words not included in class dictionary
                if word not in positive_dict:
                    positive_dict[word] = 1 / (num_positive_values + num_distinct_total_values)
                if word not in fairly_positive_dict:
                    fairly_positive_dict[word] = 1 / (num_fairly_positive_values + num_distinct_total_values)
                if word not in neutral_dict:
                    neutral_dict[word] = 1 / (num_neutral_values + num_distinct_total_values)
                if word not in fairly_negative_dict:
                    fairly_negative_dict[word] = 1 / (num_fairly_negative_values + num_distinct_total_values)
                if word not in negative_dict:
                    negative_dict[word] = 1 / (num_negative_values + num_distinct_total_values)
            return [negative_dict,fairly_negative_dict,neutral_dict,fairly_positive_dict,positive_dict]
            

            

