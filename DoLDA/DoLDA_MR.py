#!/usr/bin/python26

'''
Distributed Online Learning for Topic Models
JinYeong Bak, Dongwoo Kim, Alice Oh
http://uilab.kaist.ac.kr/research/DoLDA

This program is Python implementation of Distributed Online Learning for LDA (DoLDA).
It is based on the open-source code of online LDA (http://www.cs.princeton.edu/~mdhoffma/).
It tests on CentOS 5, Cloudera CDH3, Hadoop 0.20.2 and Python 2.6.8.
When you got troubles to run this code, please contact to Author (jy.bak@kaist.ac.kr).
'''

import dumbo
import sys
import numpy
from scipy.special import psi, gammaln, polygamma
import re
import json
from hadoop.io import SequenceFile
from hadoop.typedbytes import TypedBytesWritable

# E step
class Mapper:
    def __init__(self):
        numpy.random.seed(100000001)
        
        self._word_num = int(self.params['word_num'])
        self._meanchangethresh = float(self.params['meanchangethresh'])
        self._topic_num = int(self.params['topic_num'])
        
        # Load parameter from distributed cache
        parameter_reader = SequenceFile.Reader('./_params')
        key_class = parameter_reader.getKeyClass()
        value_class = parameter_reader.getValueClass()
        key_instance = key_class()
        value_instance = value_class()
        
        while parameter_reader.next(key_instance, value_instance):
            key_instance_str = key_instance.toString()
            if 'new_alpha' == key_instance_str:
                # For alpha
                self._alpha = value_instance.toString()
                self._alpha = numpy.fromstring(self._alpha)
                self._alpha.shape = self._topic_num
            elif 'new_lambda' == key_instance_str:
                # For lambda
                self._lambda = value_instance.toString()
                self._lambda = numpy.fromstring(self._lambda)
                self._lambda.shape = (self._topic_num, self._word_num)
            elif 'new_eta' == key_instance_str:
                # For eta
                # loading useless
                continue
            else:
                # Error
                sys.stderr.write("Something wrong in parameter_reader\n")
                sys.exit(1)
        
        parameter_reader.close()
        
        self._Elogbeta = self.dirichlet_expectation(self._lambda)
        self._expElogbeta = numpy.exp(self._Elogbeta)
        
        
    def dirichlet_expectation(self, alpha):
        """
        For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
        It comes from online LDA
        """
        if (len(alpha.shape) == 1):                 # vector
            return(psi(alpha) - psi(numpy.sum(alpha)))
        return(psi(alpha) - psi(numpy.sum(alpha, 1))[:, numpy.newaxis]) # matrix
        
        
    def e_step(self, ids, cts, gammad, expElogbetad):
        '''
        Do e step
        It comes from online LDA
        '''
        # The optimal phi_{dwk} is proportional to 
        # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
        Elogthetad = self.dirichlet_expectation(gammad)
        expElogthetad = numpy.exp(Elogthetad)
        phinorm = numpy.dot(expElogthetad, expElogbetad) + 1e-100   # scalar
        # Iterate between gamma and phi until convergence
        for it in range(0, 100):
            lastgamma = gammad
            # We represent phi implicitly to save memory and time.
            # Substituting the value of the optimal phi back into
            # the update for gamma gives this update. Cf. Lee&Seung 2001.
            gammad = self._alpha + expElogthetad * \
                numpy.dot(cts / phinorm, expElogbetad.T)    # inner product with n_1 w_1
            Elogthetad = self.dirichlet_expectation(gammad)
            expElogthetad = numpy.exp(Elogthetad)
            phinorm = numpy.dot(expElogthetad, expElogbetad) + 1e-100
            # If gamma hasn't changed much, we're done.
            meanchange = numpy.mean(abs(gammad - lastgamma))
            if (meanchange < self._meanchangethresh):
                break
        # Contribution of document d to the expected sufficient
        # statistics for the M step.
        sstats = numpy.outer(expElogthetad.T, cts/phinorm)

        return (gammad, sstats, Elogthetad)
        
        
    def __call__(self, key, value):
        '''
        Execute Map function
        
        key - each document
        value - each document content
        '''
        
        # one_doc == doc_id word_freq_all word_id:word_freq word_id:word_freq
        # one_doc == doc_id word_freq_all word_id:word_freq word_id:word_freq
        # Better code is written in online hdp!
        splitexp = re.compile(r'[ :]')
        #splitline = [int(i) for i in splitexp.split(value.strip())]
        splitline = splitexp.split(value.strip())
        doc_id = splitline[0]
        ids = [int(x) for x in splitline[2::2]]
        cts = [int(x) for x in splitline[3::2]]
        '''
        one_doc = value.split()
        doc_id = one_doc.pop(0)
        del one_doc[0]
        
        # Get document information (parsing)
        ddict = dict()
        for each_word in one_doc:
            temp_str = each_word.split(':')
            target_word_idx = int(temp_str[0])
            target_word_freq = temp_str[1]
            if (not target_word_idx in ddict):
                ddict[target_word_idx] = 0
            ddict[target_word_idx] += int(target_word_freq)
            #yield (target_word_idx, int(target_word_freq))
        ids = ddict.keys()
        cts = ddict.values()
        '''
        
        expElogbetad = self._expElogbeta[:, ids]
        gammad = 1*numpy.random.gamma(100., 1./100., self._topic_num)
        
        # E step
        (gammad, sstats, Elogthetad) = self.e_step(ids, cts, gammad, expElogbetad)
        
        # for perplexity
        phinorm = numpy.zeros(len(ids))
        Elogtheta_d = self.dirichlet_expectation(gammad)
        for i in range(0, len(ids)):
            temp = Elogtheta_d + self._Elogbeta[:, ids[i]]
            tmax = max(temp)
            phinorm[i] = numpy.log(sum(numpy.exp(temp - tmax))) + tmax
        score = numpy.sum(cts * phinorm)
        
        # for alpha update
        yield ('Elogthetad', (doc_id, Elogthetad.tostring()))
        
        # Map Output
        yield ('gammad', (doc_id, gammad.tostring()))
        yield ('sstats', (ids, sstats.tostring()))
        yield ('score', float(score))
        yield ('sum_cts', sum(cts))
        
        
# Combiner
class Combiner:
    def __init__(self):
        self._word_num = int(self.params['word_num'])
        self._topic_num = int(self.params['topic_num'])
        self._sstats = numpy.zeros((self._topic_num, self._word_num))
        self._ids_set = set()
        
    def __call__(self, key, values):
        '''
        Execute Combiner
        sum sufficient statistics
        
        key - sstats, score, sum_cts
        value - each document content
        '''
        if 'sstats' == key:
            # sstats
            for each_value in values:
                (ids, each_sstats) = each_value
                each_sstats = numpy.fromstring(each_sstats)
                each_sstats.shape = (self._topic_num, len(ids))
                self._sstats[:, ids] += each_sstats
                
                for each_ids in ids:
                    self._ids_set.add(each_ids)
            
            ids_set_list = list(self._ids_set)
            yield ('sstats_sum', (ids_set_list, self._sstats[:, ids_set_list].tostring()))
        elif 'score' == key:
            # score
            score_sum = 0
            for each_value in values:
                score_sum += each_value
            yield ('gammad', ('score_partial_sum', score_sum))
        elif 'sum_cts' == key:
            # sum_cts
            sum_cts_sum = 0
            for each_value in values:
                sum_cts_sum += each_value
            yield ('gammad', ('sum_cts_sum', sum_cts_sum))
        else:
            # etc
            for each_value in values:
                yield (key, each_value)


# M step
class Reducer:
    def dirichlet_expectation(self, alpha):
        """
        For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
        It comes from online LDA
        """
        if (len(alpha.shape) == 1):                 # vector
            return(psi(alpha) - psi(numpy.sum(alpha)))
        return(psi(alpha) - psi(numpy.sum(alpha, 1))[:, numpy.newaxis]) # matrix
    
    
    def approx_bound(self, score, sum_cts_sum):
        '''
        Compute lower bound of perplexity
        '''
        # get score
        # Should be loaded variable
        lambda_matrix = self._lambda
        Elogbeta = self._Elogbeta
        topic_num = self._topic_num
        minibatch_size = self._minibatch_size
        Elogtheta = self.dirichlet_expectation(self.gamma)
        word_num = self._word_num
        document_num = self._document_num
        
        # calculate score
        alpha_vec = self._alpha
        eta_vec = self._eta
        
        score += numpy.sum((alpha_vec - self.gamma) * Elogtheta)
        score += numpy.sum(gammaln(self.gamma) - gammaln(alpha_vec))
        #score += sum(gammaln(alpha_vec * topic_num) - gammaln(numpy.sum(self.gamma, 1)))
        # Changed from oLDA, beacuse we use alpha as vector, not single value
        score += sum(gammaln(numpy.sum(alpha_vec)) - gammaln(numpy.sum(self.gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * document_num / minibatch_size
        # E[log p(beta | eta) - log q (beta | lambda)]
        score += numpy.sum((eta_vec - lambda_matrix) * Elogbeta)
        score += numpy.sum(gammaln(lambda_matrix) - gammaln(eta_vec))       
        #score += numpy.sum(gammaln(eta_vec * word_num) - gammaln(numpy.sum(lambda_matrix, 1)))
        # Changed from oLDA, beacuse we use eta as vector, not single value
        score += numpy.sum(gammaln(numpy.sum(eta_vec)) - gammaln(numpy.sum(lambda_matrix, 1)))
        
        perwordbound = score * minibatch_size / (document_num * sum_cts_sum)
        perwordbound_exp = numpy.exp(-perwordbound)
        return perwordbound_exp
        
    def __init__(self):
        self._word_num = int(self.params['word_num'])
        self._document_num = int(self.params['document_num'])
        self._minibatch_size = int(self.params['minibatch_size'])
        self._meanchangethresh = float(self.params['meanchangethresh'])
        self._topic_num = int(self.params['topic_num'])
        
        self._tau0 = float(self.params['tau0'])
        self._updatect = float(self.params['updatect'])
        self._kappa = float(self.params['kappa'])
        
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        
        # Load parameter from distributed cache
        parameter_reader = SequenceFile.Reader('./_params')
        key_class = parameter_reader.getKeyClass()
        value_class = parameter_reader.getValueClass()
        key_instance = key_class()
        value_instance = value_class()
        
        while parameter_reader.next(key_instance, value_instance):
            key_instance_str = key_instance.toString()
            if 'new_alpha' == key_instance_str:
                # For alpha
                self._alpha = value_instance.toString()
                self._alpha = numpy.fromstring(self._alpha)
                self._alpha.shape = self._topic_num
            elif 'new_lambda' == key_instance_str:
                # For lambda
                self._lambda = value_instance.toString()
                self._lambda = numpy.fromstring(self._lambda)
                self._lambda.shape = (self._topic_num, self._word_num)
            elif 'new_eta' == key_instance_str:
                # For eta
                self._eta = value_instance.toString()
                self._eta = numpy.fromstring(self._eta)
                self._eta.shape = self._word_num
            else:
                # Error
                sys.stderr.write("Something wrong in parameter_reader\n")
                sys.exit(1)
        
        parameter_reader.close()
        
        self._Elogbeta = self.dirichlet_expectation(self._lambda)
        self._expElogbeta = numpy.exp(self._Elogbeta)
        
        # initialize sstats
        self.sstats = numpy.zeros((self._topic_num, self._word_num))
        self.gamma = numpy.zeros((self._minibatch_size, self._topic_num))
        
    def __call__(self, key, values):
        '''
        Execute Reducer
        Do M step
        
        key - sstats, score, sum_cts
        value - each document content
        '''
        # Get this from distributed cache
#        topic_num = self._topic_num
#        _eta = 1./topic_num
        rhot = self._rhot
        document_size = self._document_num
        mini_batch = self._minibatch_size
        
        if 'sstats_sum' == key:
            # sstats_sum
            for each_value in values:
                (ids, each_sstats) = each_value
                each_sstats = numpy.fromstring(each_sstats)
                each_sstats.shape = (self._topic_num, len(ids))
                self.sstats[:, ids] += each_sstats
            
            # Get new lambda
            self.sstats = self.sstats * self._expElogbeta
            self.new_lambda = self._lambda * (1. - rhot) + \
                    rhot * (self._eta + document_size * self.sstats / mini_batch)
            
            # outputs computed lambda
            yield (('parameters', 'new_lambda'), self.new_lambda.tostring())
        elif 'gammad' == key:
            # gammad
            score_sum = 0
            sum_cts_sum = 0
            
            doc_idx = 0
            
            for each_value in values:
                (each_value_key, each_value_value) = each_value
                if 'score_partial_sum' == each_value_key:
                    # score_partial_sum
                    score_sum += each_value_value
                elif 'sum_cts_sum' == each_value_key:
                    # sum_cts_sum
                    sum_cts_sum += each_value_value
                else:
                    # gammad
                    gammad = numpy.fromstring(each_value_value)
                    gammad.shape = (self._topic_num)
                    self.gamma[doc_idx, :] = gammad
                    doc_idx += 1
            
            perwordbound_exp = self.approx_bound(score_sum, sum_cts_sum)
            
            yield (('infor', 'perplexity'), str(perwordbound_exp))
#            yield (('infor', 'rhot'), self._rhot)
#            yield (('infor', 'updatect'), self._updatect)
        elif 'Elogthetad' == key:
            # Update alpha
            g_left_term = self.dirichlet_expectation(self._alpha)
            q_inv = -1. / polygamma(1, self._alpha)
            z_inv = 1. / polygamma(1, numpy.sum(self._alpha))
            sum_s = numpy.zeros(self._topic_num)
            denom = z_inv + numpy.sum(q_inv)
            
            for each_value in values:
                (doc_id, Elogthetad) = each_value
                Elogthetad = numpy.fromstring(Elogthetad)
                Elogthetad.shape = (self._topic_num)
                g_ = Elogthetad - g_left_term
                
                sum_s += (g_ - ((numpy.sum(g_ * q_inv)) / denom))
                
            self.new_alpha = self._alpha - sum_s * q_inv * self._rhot / self._minibatch_size
            
            # Update eta
            g_left_term = self.dirichlet_expectation(self._eta) * self._topic_num
            q_inv = -1. / (self._topic_num * polygamma(1, self._eta))
            z_inv = 1. / (self._topic_num * polygamma(1, numpy.sum(self._eta)))
            g_ = numpy.sum(self.dirichlet_expectation(self._lambda), axis=0) - g_left_term
            sum_s = numpy.zeros(self._topic_num)
            denom = z_inv + numpy.sum(q_inv)
            
            self.new_eta = self._eta - (g_ - ((numpy.sum(g_ * q_inv)) / denom)) * q_inv * self._rhot
            
            # Output
            yield(('parameters', 'new_alpha'), self.new_alpha.tostring())
            yield(('parameters', 'new_eta'), self.new_eta.tostring())
            
        else:
            # others
            # key is doc_id
            for each_value in values:
                yield ((key, key), each_value)
        
# main function start
if __name__ == "__main__":    
    # job execute
    job = dumbo.Job()
    job.additer(Mapper, Reducer, combiner=Combiner)
    job.run()
    
