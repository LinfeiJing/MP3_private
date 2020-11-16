import os
os.system('pip install metapy')
import numpy as np
import math
import metapy
import re



def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)
        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        # #############################
        with open(self.documents_path) as file:
            content = file.readlines()

        for i in range(len(content)):
            doc = metapy.index.Document()
            doc.content(content[i])
            tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
            tok.set_content(doc.content())
            tokens_temp = [token for token in tok]

            if i < 100:
                self.documents.append(tokens_temp[1:])
            else:
                self.documents.append(tokens_temp)

            self.number_of_documents = self.number_of_documents + 1

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        # your code here
        # #############################
        for w in open(self.documents_path).read().split():
            if w not in self.vocabulary and w != '0' and w != '1':
                self.vocabulary.append(w)

        self.vocabulary_size = len(self.vocabulary)  # total?

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
        with open(self.documents_path) as file:
            content = file.readlines()
        self.term_doc_matrix = np.zeros((self.number_of_documents, self.vocabulary_size))

        for i in range(self.number_of_documents):
            for j in range(self.vocabulary_size):
                self.term_doc_matrix[i][j] = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(self.vocabulary[j]), content[i]))

        # ############################
        



    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here
        self.document_topic_prob = normalize(np.random.rand(self.number_of_documents, number_of_topics))
        self.topic_word_prob = normalize(np.random.rand(number_of_topics, len(self.vocabulary)))

        # ############################

        pass    # REMOVE THIS
        

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        
        # ############################
        # your code here
        for i in range(self.number_of_documents):
            for j in range(len(self.topic_prob[0])):
                for k in range(self.vocabulary_size):
                    self.topic_prob[i][j][k] = (self.document_topic_prob[i][j] * self.topic_word_prob[j][k]) / np.sum(self.document_topic_prob[i, :] * self.topic_word_prob[:, k])

        # ############################

        # pass    # REMOVE THIS
            

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")
        
        # update P(w | z)
        
        # ############################
        # your code here
        for i in range(len(self.document_topic_prob)):
            for j in range(len(self.document_topic_prob[0])):
                den = 0
                for k in range(len(self.document_topic_prob[0])):
                    den = den + sum(self.term_doc_matrix[i, :] * self.topic_prob[i, k, :])
                self.document_topic_prob[i][j] = sum(self.term_doc_matrix[i, :] * self.topic_prob[i, j, :]) / den

        # ############################

        
        # update P(z | d)

        # ############################
        # your code here
        for i in range(number_of_topics):
            for j in range(len(self.topic_word_prob[0])):
                self.topic_word_prob[i][j] = sum(self.term_doc_matrix[:, j] * self.topic_prob[:, i, j])
        self.topic_word_prob = normalize(self.topic_word_prob)

        # ############################
        
        # pass    # REMOVE THIS


    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################
        # your code here
        temp = np.matmul(self.document_topic_prob, self.topic_word_prob)
        prob = 0
        for i in range(len(temp)):
            for j in range(len(temp[0])):
                prob = prob + self.term_doc_matrix[i, j] * np.log(temp[i, j])

        return prob
        # ############################
        


    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            # your code here
            self.expectation_step()
            # m-step
            self.maximization_step(number_of_topics)
            temp2 = self.calculate_likelihood(number_of_topics)
            print(temp2)
            self.likelihoods.append(temp2)
            if iteration == 0:
                continue

            else:
                if self.likelihoods[iteration] - self.likelihoods[iteration - 1] < epsilon:
                    break
            # ############################

            pass    # REMOVE THIS



def main():
    documents_path = 'data/test.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 50
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)


if __name__ == '__main__':
    main()
