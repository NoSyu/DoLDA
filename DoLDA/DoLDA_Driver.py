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

import sys
import subprocess
import os
import numpy
from scipy.special import gammaln, psi
import re
import ctypedbytes
from hadoop.io import SequenceFile
from hadoop.typedbytes import TypedBytesWritable

def file_len(fname):
    '''
    Return file length of fname
    
    fname - File path. String
    
    It comes from
    http://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
    '''
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def init_parameters(topic_num, word_num, hadoop_hdfs_root):
    '''
    Initialize parameters, alpha, lambda and eta
    '''
    # parameter initialized    
    numpy.random.seed(100000001)
    
    # file setting
    parameter_target_filename = 'parameters_for_0.txt'
    
    writer = SequenceFile.createWriter(parameter_target_filename, TypedBytesWritable, TypedBytesWritable)
    
    # For alpha
    _alpha = numpy.zeros(topic_num) + 1./topic_num
    
    output_key_a = TypedBytesWritable()
    output_value_a = TypedBytesWritable()

    output_key_a.set('new_alpha')
    output_value_a.set(_alpha.tostring())

    writer.append(output_key_a, output_value_a)
    
    # For lambda
    _lambda = 1*numpy.random.gamma(100., 1./100., (topic_num, word_num))
    
    output_key_l = TypedBytesWritable()
    output_value_l = TypedBytesWritable()

    output_key_l.set('new_lambda')
    output_value_l.set(_lambda.tostring())

    writer.append(output_key_l, output_value_l)
    
    # For eta
    _eta = numpy.zeros(word_num) + 1./topic_num
    
    output_key_e = TypedBytesWritable()
    output_value_e = TypedBytesWritable()

    output_key_e.set('new_eta')
    output_value_e.set(_eta.tostring())

    writer.append(output_key_e, output_value_e)

    writer.close()
    
    subprocess.call("hadoop dfs -copyFromLocal " + parameter_target_filename + " " + hadoop_hdfs_root, shell=True, stdout=file(os.devnull, "w"))
    os.remove(parameter_target_filename)
    
    return parameter_target_filename
    
    
def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    It comes from online LDA
    """
    if (len(alpha.shape) == 1):                 # vector
        return(psi(alpha) - psi(numpy.sum(alpha)))
    return(psi(alpha) - psi(numpy.sum(alpha, 1))[:, numpy.newaxis]) # matrix
        

# main function start
if __name__ == "__main__":
    # input check
    if len(sys.argv) < 11:
        sys.exit('Usage: %s word_file_path document_file_path topic_num minibatch_size tau0 kappa num_mapper num_reducer hadoop_hdfs_root hadoop_library_path' % sys.argv[0])
        
    # input setting
    word_file_path = sys.argv[1]
    document_file_path = sys.argv[2]
    topic_num = int(sys.argv[3])
    minibatch_size = int(sys.argv[4])
    tau0 = float(sys.argv[5])
    kappa = float(sys.argv[6])
    num_mapper = int(sys.argv[7])
    num_reducer = int(sys.argv[8])
    hadoop_hdfs_root = sys.argv[9]
    hadoop_lib_path = sys.argv[10]   # example: '/usr/lib/hadoop-0.20'    # CDH3
    python_bin_path = sys.argv[11]  # example: '/usr/bin/python26'        # CentOS 5
    
    # parameter setting
    word_num = file_len(word_file_path)
    document_num = file_len(document_file_path)
    
    # from online lda code
    meanchangethresh = 0.001
    updatect = 0
    
    minibatch_filename = 'BOW_LDA_minibatch_%s.txt' % minibatch_size
    
    # divide the document
    # loop
    doc_loop_count = (document_num / minibatch_size) + 1
    BOW_file = open(document_file_path, 'r')
    
    job_execute_command_template = "dumbo start DoLDA_MR.py -input %s/%s -output %s/output_%s -python %s -memlimit 4294967296 -hadoop /usr -hadooplib %s -outputformat sequencefile -nummaptasks %d -getpath yes -file DoLDA_MR.py -param word_num=%s -param document_num=%s -param minibatch_size=%s -param meanchangethresh=%s -param topic_num=%s -param tau0=%s -param updatect=%s -param kappa=%s -libjar feathers.jar -hadoopconf stream.recordreader.compression=gzip -numreducetasks %d -libegg ctypedbytes-0.1.9-py2.6-linux-x86_64.egg -libegg Hadoop-0.1-py2.6.egg -cmdenv PYTHON_EGG_CACHE=/tmp/eggcache -cachefile %s/%s#_params"
    
    #doc_loop_count = 2  # For debugging
    doc_loop_count_m1 = doc_loop_count - 1
    for updatect in range(0, doc_loop_count):
        # generate new target document file
        target_file = open(minibatch_filename, 'w')
        
        if doc_loop_count_m1 == updatect:
            # last docs
            docs_num = 0
            for iter in range(0, minibatch_size):
                one_doc = BOW_file.readline()
                #print >>target_file, one_doc
                if one_doc:
                    # one_doc is existed
                    docs_num += 1
                    target_file.write(one_doc)
                else:
                    # reach the end line
                    break
            minibatch_size = docs_num
        else:        
            for iter in range(0, minibatch_size):
                one_doc = BOW_file.readline()
                target_file.write(one_doc)
            
        target_file.close()
        
        # delete and upload input file
        subprocess.call("hadoop dfs -rm %s/%s" % (hadoop_hdfs_root, minibatch_filename), shell=True, stdout=file(os.devnull, "w"))
        subprocess.call("hadoop dfs -copyFromLocal %s %s/" % (minibatch_filename, hadoop_hdfs_root), shell=True, stdout=file(os.devnull, "w"))
        
         # parameter lambda, alpha, eta
        if 0 == updatect:
            lambda_target_filename = init_parameters(topic_num, word_num, hadoop_hdfs_root)
        else:
            lambda_target_filename = 'output_%d/parameters/parameters' % (updatect-1)
     
        # job execute
        job_execute_command = job_execute_command_template % (hadoop_hdfs_root, minibatch_filename, hadoop_hdfs_root, str(updatect), python_bin_path, hadoop_lib_path, num_mapper, str(word_num), str(document_num), str(minibatch_size), str(meanchangethresh), str(topic_num), str(tau0), str(updatect), str(kappa), num_reducer, hadoop_hdfs_root, lambda_target_filename)
        subprocess.call(job_execute_command, shell=True, stdout=file(os.devnull, "w"))
        
        # job finish
        
    BOW_file.close()
