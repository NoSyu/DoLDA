# Input of this script
# word_file_path document_file_path topic_num minibatch_size tau0 kappa num_mapper hadoop_hdfs_root
# Example
# sh DoLDA.sh ../Twitter_Conversation/Voca_TC.txt ../Twitter_Conversation/BOW_TC.txt 100 16384 1024 0.7 3 /user/hadoop_usr

hadoop dfs -rmr $8/output*
hadoop dfs -rmr $8/lambda*
rm -rf ./output*
time python26 DoLDA_Driver.py $1 $2 $3 $4 $5 $6 $7 $8
hadoop dfs -copyToLocal $8/output* ./