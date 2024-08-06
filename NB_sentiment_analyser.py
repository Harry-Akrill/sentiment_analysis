# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
from __future__ import division
import argparse
import nltk
from nltk.sentiment.util import *
import pandas as pd
import numpy as np
from sentence import Sentence
from sentence_set import SentenceSet
import matplotlib.pyplot as plt
from preprocess import Preprocess


"""
IMPORTANT, modify this part with your details
"""
USER_ID = "aca19haa" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args


def main():
    nltk.download()
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    """
    ADD YOUR CODE HERE
    Create functions and classes, using the best practices of Software Engineering
    """

    #You need to change this in order to return your macro-F1 score for the dev set
    f1_score = 0

    train_csv = pd.read_csv(training, sep='\t')
    dev_csv = pd.read_csv(dev, sep='\t')
    test_csv = pd.read_csv(test, sep='\t')

    #setting training data to sentences
    np_train_csv = train_csv.to_numpy()
    training_sentence_list = list()
    for i in np_train_csv: # preprocessing
        process = Preprocess(i[1],features,True)
        phrase = process.processed_phrase
        sentence = Sentence(i[0],phrase,i[2],number_classes)
        training_sentence_list.append(sentence)

    #setting development data to sentences
    np_dev_csv = dev_csv.to_numpy()
    dev_sentence_list = list()
    #negation_dict = dict() # Failed negation implementation
    for i in np_dev_csv:
        phrase = Preprocess(i[1],features,False).processed_phrase
        
        #split_phrase = phrase.split(" ")
        # negation = mark_negation(split_phrase)
        # print(negation)
        # if any("_NEG" in word for word in negation):
        #     negation_dict[i[0]] = 1
        
        sentence = Sentence(i[0],phrase,i[2],number_classes)
        dev_sentence_list.append(sentence) 

    #setting test data to sentences
    np_test_csv = test_csv.to_numpy()
    test_sentence_list = list()
    for i in np_test_csv:
        phrase = Preprocess(i[1],features,False).processed_phrase
        sentence = Sentence(i[0],phrase,1,number_classes)
        test_sentence_list.append(sentence)

    #compute class counts
    class_counts = []
    if number_classes == 3:
        negative_count = len([x for x in training_sentence_list if x.sentiment_class == 0])
        neutral_count = len([x for x in training_sentence_list if x.sentiment_class == 1])
        positive_count = len([x for x in training_sentence_list if x.sentiment_class == 2])
        class_counts = [negative_count,neutral_count,positive_count]
    else:
        negative_count = len([x for x in training_sentence_list if x.sentiment_class == 0])
        fairly_negative_count = len([x for x in training_sentence_list if x.sentiment_class == 1])
        neutral_count = len([x for x in training_sentence_list if x.sentiment_class == 2])
        fairly_positive_count = len([x for x in training_sentence_list if x.sentiment_class == 3])
        positive_count = len([x for x in training_sentence_list if x.sentiment_class == 4])
        class_counts = [negative_count,fairly_negative_count,neutral_count,fairly_positive_count,
                        positive_count]
    
    prior_probabilites = compute_prior_probability(class_counts,number_classes)
    
    #Create SentenceSet object so feature likelihoods can be obtained
    training_sentence_set = SentenceSet(training_sentence_list,number_classes,features)

    posteriors_dev = calculate_posteriors(training_sentence_set,prior_probabilites,dev_sentence_list,number_classes)
    posteriors_test = calculate_posteriors(training_sentence_set,prior_probabilites,test_sentence_list,number_classes)

    # for sentenceid in posteriors_dev: # Failed negation implementation
    #     if sentenceid in negation_dict:
    #         if posteriors_dev[sentenceid] == 0:
    #             posteriors_dev[sentenceid] = 2
    #         if posteriors_dev[sentenceid] == 2:
    #             posteriors_dev[sentenceid] = 0

    f1_score = calculate_macro_f1_score(posteriors_dev,dev_sentence_list,number_classes)

    if output_files == True:
        save_predictions(posteriors_dev,posteriors_test,number_classes)
    if confusion_matrix == True:
        display_confusion_matrices(posteriors_dev,dev_sentence_list)
    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    #print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

def display_confusion_matrices(predicted,actual):
    """
    Displays a confusion matrix between the predicted and actual sentence sentiment values

    Parameters:
    ----------
    predicted : dictionary
            dictionary of sentence id to predicted sentiment values
    actual : dictionary
            dictionary of sentience id to actual sentiment values
    
    """
    predicted_values = pd.Series(predicted.values(), name='Predicted')
    actual_values = list()
    for i in actual:
        actual_values.append(i.sentiment_class)
    actual_values = pd.Series(actual_values, name='Actual')
    confusion_matrix = pd.crosstab(actual_values,predicted_values) # Create confusion matrix
    plt.matshow(confusion_matrix, cmap=plt.cm.Greens) # Create figure to display
    plt.colorbar()
    columns = confusion_matrix.columns
    plt.ylabel(confusion_matrix.index.name)
    plt.xlabel(columns.name)
    plt.show()

def save_predictions(dev_predictions,test_predictions,num_classes):#
    """
    Saves predictions for test and development sets

    Parameters:
    ----------
    dev_predictions : dictionary
                dictionary of sentence id to predicted sentiment values for development set
    test_predictions : dictionary
                dictionary of sentence id to predicted sentiment values for test set
    num_classes : Integer
                The specified number of classes to be used in the classifier
    """
    dic_dev = dict()
    dic_test = dict()
    dic_dev = {"SentenceID" : dev_predictions.keys(), "Sentiment" : dev_predictions.values()}
    dic_test = {"SentenceID" : test_predictions.keys(), "Sentiment" : test_predictions.values()}
    dev_data = pd.DataFrame(dic_dev)
    test_data = pd.DataFrame(dic_test)
    dev_data.to_csv('dev_predictions_' + str(num_classes) + 'classes_' + USER_ID + '.tsv', sep ="\t", index=False)
    test_data.to_csv('test_predictions_' + str(num_classes) + 'classes_'+ USER_ID + '.tsv', sep ="\t", index=False)

def compute_prior_probability(class_counts,num_classes): # returns array of prior probabilities
    """
    Computes prior probabilites for classes

    Parameters :
    ----------
    class_counts : Array of integers
                Array of class counts for negative,neutral, and positive classes for 3 classes
                Includes fairly positive/negative for 5 classes
    num_classes : Integer
                The specified number of classes to be used in the classifier
    """
    if num_classes == 3:
        return [(class_counts[0]/sum(class_counts)),(class_counts[1]/sum(class_counts)),(class_counts[2]/sum(class_counts))]
    if num_classes == 5:
        return [(class_counts[0]/sum(class_counts)),(class_counts[1]/sum(class_counts)),(class_counts[2]/sum(class_counts)),
                (class_counts[3]/sum(class_counts)),(class_counts[4]/sum(class_counts))]

def calculate_posteriors(trained_set,prior_probabilities,test_set,num_classes):
    """
    Calculates posterior probabilities and returns mapping of sentence id to chosen sentiment class

    Parameters :
    ----------
    trained_set : list
                list of Sentence objects for each sentence in training set
    prior_probabilites : Array
                Prior probabilities for each class in classifier
    test_set : list
                List of sentences to be tested on
    num_classes : Integer
                Chosen number of classes for classifier
    """
    id_to_posterior = dict()
    if num_classes == 3:
        negative_prior = prior_probabilities[0]
        neutral_prior = prior_probabilities[1]
        positive_prior = prior_probabilities[2]

        negative_likelihoods = trained_set.f_likelihoods[0]
        neutral_likelihoods = trained_set.f_likelihoods[1]
        positive_likelihoods = trained_set.f_likelihoods[2]

        for sentence in test_set:
            phrase = sentence.phrase
            sentence_id = sentence.sentence_id
            posteriors = np.array([0,0,0])
            posteriors = posteriors.astype('float64')
            count = 0
            for word in phrase:
                if word in (negative_likelihoods or neutral_likelihoods or positive_likelihoods): # Check if in ANY class
                    count += 1
                    #calculate negative posteriors
                    if word in negative_likelihoods:
                        if posteriors[0] == 0 and count == 1:
                            posteriors[0] = negative_likelihoods[word]
                        else:
                            posteriors[0] = posteriors[0] * (negative_likelihoods)[word] # Posterior calculation
                    else:
                        posteriors[0] = 0

                    #calculate neutral posteriors
                    if word in neutral_likelihoods:
                        if posteriors[1] == 0 and count == 1:
                            posteriors[1] = (neutral_likelihoods)[word]
                        else:
                            posteriors[1] = posteriors[1] * (neutral_likelihoods)[word]
                    else:
                        posteriors[1] = 0

                    #calculate positive posteriors
                    if word in positive_likelihoods:    
                        if posteriors[2] == 0 and count == 1:
                            posteriors[2] = (positive_likelihoods)[word]
                        else:
                            posteriors[2] = posteriors[2] * (positive_likelihoods)[word]
                    else:
                        posteriors[2] = 0

            posteriors[0] = posteriors[0] * negative_prior
            posteriors[1] = posteriors[1] * neutral_prior
            posteriors[2] = posteriors[2] * positive_prior
            id_to_posterior[sentence_id] = np.argmax(posteriors)
        return id_to_posterior
    
    if num_classes == 5:
        negative_prior = prior_probabilities[0]
        fairly_negative_prior = prior_probabilities[1]
        neutral_prior = prior_probabilities[2]
        fairly_positive_prior = prior_probabilities[3]
        positive_prior = prior_probabilities[4]

        negative_likelihoods = trained_set.f_likelihoods[0]
        fairly_negative_likelihoods = trained_set.f_likelihoods[1]
        neutral_likelihoods = trained_set.f_likelihoods[2]
        fairly_positive_likelihoods = trained_set.f_likelihoods[3]
        positive_likelihoods = trained_set.f_likelihoods[4]

        for sentence in test_set:
            phrase = sentence.phrase
            sentence_id = sentence.sentence_id
            posteriors = np.array([0,0,0,0,0])
            posteriors = posteriors.astype('float64')
            count = 0
            for word in phrase:
                if word in (negative_likelihoods or fairly_negative_likelihoods or neutral_likelihoods or
                            fairly_positive_likelihoods or positive_likelihoods):
                    count += 1
                    if word in negative_likelihoods:
                        if posteriors[0] == 0 and count == 1:
                            posteriors[0] = negative_likelihoods[word]
                        else:
                            posteriors[0] = posteriors[0] * negative_likelihoods[word]
                    else:
                        posteriors[0] = 0 # dont add it if it doesnt appear in the training set

                    if word in fairly_negative_likelihoods:
                        if posteriors[1] == 0 and count == 1:
                            posteriors[1] = fairly_negative_likelihoods[word]
                        else:
                            posteriors[1] = posteriors[1] * fairly_negative_likelihoods[word]
                    else:
                        posteriors[1] = 0
                    
                    if word in neutral_likelihoods:    
                        if posteriors[2] == 0 and count == 1:
                            posteriors[2] = neutral_likelihoods[word]
                        else:
                            posteriors[2] = posteriors[2] * neutral_likelihoods[word]
                    else:
                        posteriors[2] = 0

                    if word in fairly_positive_likelihoods:    
                        if posteriors[3] == 0 and count == 1:
                            posteriors[3] = fairly_positive_likelihoods[word]
                        else:
                            posteriors[3] = posteriors[3] * fairly_positive_likelihoods[word]
                    else:
                        posteriors[3] = 0

                    if word in positive_likelihoods:    
                        if posteriors[4] == 0 and count == 1:
                            posteriors[4] = positive_likelihoods[word]
                        else:
                            posteriors[4] = posteriors[4] * positive_likelihoods[word]
                    else:
                        posteriors[4] = 0

            posteriors[0] = posteriors[0] * negative_prior
            posteriors[1] = posteriors[1] * fairly_negative_prior
            posteriors[2] = posteriors[2] * neutral_prior
            posteriors[3] = posteriors[3] * fairly_positive_prior
            posteriors[4] = posteriors[4] * positive_prior
            id_to_posterior[sentence_id] = np.argmax(posteriors)
        return id_to_posterior
                       
def calculate_macro_f1_score(posteriors,actual_sentences,num_classes):
    """
    Calculates macro f1 score

    Parameters :
    ----------
    posteriors : dictionary
                Dictionary containing mapping of sentence id to chosen sentiment value according to posteriors
    actual_sentences : list 
                List of sentence objects with correct sentiment value
    num_classes : Integer
                Chosen number of classes for the classifier

    """
    if num_classes == 3:
        f1_scores = np.array([0,0,0]).astype('float64') # Set array type to float to account for large decimals
        for i in range(3):
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            count = 0
            for j in posteriors:
                if posteriors[j] == i and actual_sentences[count].sentiment_class == i:
                    true_positives += 1
                    count += 1
                elif posteriors[j] == i and actual_sentences[count].sentiment_class != i:
                    false_positives += 1
                    count += 1
                elif posteriors[j] != i and actual_sentences[count].sentiment_class == i:
                    false_negatives += 1
                    count += 1
                else:
                    count += 1
            f1_scores[i] = (2 * true_positives) / ((2 * true_positives) + false_positives + false_negatives)
        f1_macro = np.mean(f1_scores)
        return f1_macro

    if num_classes == 5:
        f1_scores = np.array([0,0,0,0,0]).astype('float64')
        for i in range(5):
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            count = 0
            for j in posteriors:
                if posteriors[j] == i and actual_sentences[count].sentiment_class == i:
                    true_positives += 1
                    count += 1
                elif posteriors[j] == i and actual_sentences[count].sentiment_class != i:
                    false_positives += 1
                    count += 1
                elif posteriors[j] != i and actual_sentences[count].sentiment_class == i:
                    false_negatives += 1
                    count += 1
                else:
                    count += 1
            f1_scores[i] = (2 * true_positives) / ((2 * true_positives) + false_positives + false_negatives)
        f1_macro = np.mean(f1_scores)
        return f1_macro         

if __name__ == "__main__":
    main()

#py NB_sentiment_analyser.py moviereviews/train.tsv moviereviews/dev.tsv moviereviews/test.tsv -classes 3