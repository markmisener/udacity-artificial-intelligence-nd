import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant
    """

    def select(self):
        """ select based on n_constant value
        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore")

        # initialize best_score and best_model
        best_score, best_model  = float("inf"), None

        # determine the lowest possible bic score and model
        for n_components in range(self.min_n_components, self.max_n_components):
            try:
                # create the base model
                model = self.base_model(n_components)
                # calculate logL
                logL = model.score(self.X, self.lengths)
                # determine number of features
                n_features = self.X.shape[1]
                # calculate number of data points
                n_points = n_components * (n_components - 1) + 2 * n_features * n_components
                # calculate logN
                logN = np.log(self.X.shape[0])
                # calculate bic score
                # BIC = -2 * logL + p * logN
                bic = -2 * logL + n_points * logN
                # set best_score and best_model if this is the lowest score so far
                if bic < best_score:
                    best_score, best_model = bic, model
            except Exception as e:
                continue

        return best_model if best_model else self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        """ select the best model for self.this_word based on
        DIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore")

        # create two lists to hold model scores and logs
        scores, logs = [], []

        try:
            for n_component in self.n_components:
                # create model
                model = self.base_model(n_component)
                # append model score to logs list
                logs.append(model.score(self.X, self.lengths))
            # sum all logs
            sum_logs = sum(logs)
            # get length of components
            m = len(self.n_components)
            for each_log in logs:
                # calculate dic score
                # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
                anti_likelihood = (sum_logs - each_log) / (m - 1)
                scores.append(each_log - anti_likelihood)
        except Exception as e:
            pass
        # determine best score and return corresponding model
        best_score = self.n_components[np.argmax(scores)] if scores else self.n_constant
        return self.base_model(best_score)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def select(self):
        warnings.filterwarnings("ignore")

        # create list to store mean scores
        mean_scores = []

        # set split method
        split_method = KFold()

        try:
            for n_component in self.n_components:
                # create model
                model = self.base_model(n_component)
                # create list to store calculated model mean scores
                fold_scores = []
                for _, test in split_method.split(self.sequences):
                    # calculate test sequences
                    test_X, test_length = combine_sequences(test, self.sequences)
                    # append model score to fold scores list
                    fold_scores.append(model.score(test_X, test_length))

                # append mean of all fold scores to mean scores list
                mean_scores.append(np.mean(fold_scores))
        except Exception as e:
            pass

        # determine best score and return corresponding model
        best_score = self.n_components[np.argmax(mean_scores)] if mean_scores else self.n_constant
        return self.base_model(best_score)
