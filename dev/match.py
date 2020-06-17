#-*- coding:utf-8 -*-

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import collections
import copy 
import numpy as np  
import pandas as pd 
import traceback
import logging
import os 
import pickle 
import faiss 
import random
import re
import itertools
import json 


PATH = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(filename=os.path.join(PATH,'FLAGS.logfile'),filemode='a',
format='%(asctime)s %(name)s:%(levelname)s:%(message)s',datefmt="%d-%m-%Y %H:%M:%S",
level=logging.DEBUG)

##############
# data prepare.py
# 
#
# 
############## 




#### paras.py
user_count = 100000
item_count = 50000
category_count = 3000
tags_count = 100

item_his_max_len =15
lr =0.0001
epoches = 5
batch_size = 128

dnn_layers =3
dnn_nums =[200,40,64]
nums_sampled=100 #??


event_name = 'click'
start_date =''
end_date = ''
start_time = ''
end_time = ''

emb_dims = 64
model_save_path ='model'
items_matrix_save_path = 'matrix' 


'''
# feature
#item user 分别做个索引文件, time加进去特征 
#
# user( id,agenda,) ,itemhis( [cateid:iid] ,[cateid:iid],...), 
#
# 
userid i1 i2
userid i1i2 i3   是不是限制下 skipnums

'''
def get_data_from_csv(path):
    dataframe = pd.read_csv(path)
    #主演,标题,
    features = ['distinct_id', 'item_id']
    feature_max_idx = {}

    for feature in features:
        lbe = LabelEncoder()

        dataframe[feature] = lbe.fit_transform(dataframe[feature]) + 1
        feature_max_idx[feature] = dataframe[feature].max() + 1
    
    return dataframe





#保存一个 category+tags 2000 个把,,字典索引
#tag为空的 选择  cate ,但tag 很快就有新的了,

def compute_index_tags(df):
    data = copy.deepcopy(df)
    for index,v in data.iterrows():
        #print(v['tags'],type(v['tags']),str(v['tags']))
        if 'nan' in str(v['tags']):
            v['tags']=v['category']
        #v['tags'] = re.split('[\：|\~|\;|\，|\、|\t|\s]',v['tags'])
        vv =  re.split('[\：|\~|\;|\，|\、|\t|\s]',v['tags'])
       # df.iloc[index]['tags'] = vv if len(vv)<2 else vv[1]
        data.loc[index,'tags'] = vv if len(vv)<2 else vv[1]

    cat_index =pickle.load(open("/home/tv/Documents/未来电视-数据流代码/未来电视-数据流代码/cate_tag_dict.pkl",'rb'))

    def change(row):
        row['category'],row['tags'] = cat_index[row['category']],cat_index[row['tags']]
        return row
    #df['category'].apply(lambda row:cat_index[row],axis=1)
    data['category'] = data.apply(lambda row:cat_index[row['category']],axis=1)
    data['tags'] = data.apply(lambda row:cat_index[row['tags']],axis=1 )
    
    data.to_csv('/home/tv/Downloads/query_result_desc_new.csv')
    return data 




def gen_data_set(data, negsample=0):

    data.sort_values("event_time", inplace=True)
    item_ids = data['item_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('distinct_id')):
        pos_list = hist['item_id'].tolist()
        cate_list = hist['category'].tolist()
        tags_list = hist['tags'].tolist()
        #decs_list = hist['desc'].tolist()



        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set,size=len(pos_list)*negsample,replace=True)
        for i in range(1, len(pos_list)):
            hist_i,hist_c,hist_t = pos_list[:i],cate_list[:i],tags_list[:i]

            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist_i[::-1], pos_list[i], 1, len(hist_i[::-1]),hist_c[::-1],hist_t[::-1]))
                for negi in range(negsample):
                    train_set.append((reviewerID, hist_i[::-1], neg_list[i*negsample+negi], 0,len(hist_i[::-1])))
            else:
                test_set.append((reviewerID, hist_i[::-1], pos_list[i],1,len(hist_i[::-1]),hist_c[::-1],hist_t[::-1]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    #print(len(train_set[0]),len(test_set[0]))

    return train_set,test_set




def gen_model_input(train_set,seq_max_len):

    train_uid = np.array([line[0] for line in train_set])
    train_his_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])
    train_cate = [line[5] for line in train_set]
    train_tag = [line[6] for line in train_set]


    train_his_seq_pad = pad_sequences(train_his_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_cate_seq_pad = pad_sequences(train_cate, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    
    train_tag_seq_pad = pad_sequences(train_tag, maxlen=seq_max_len, padding='post', truncating='post', value=0)


    train_model_input = {"user_id_ph": train_uid, "item_id_ph": train_iid, "his_item_seq_ph": train_his_seq_pad,
                         "his_len_ph": train_hist_len,"his_cate_seq_ph":train_cate_seq_pad,"his_tag_seq_ph":train_tag_seq_pad}

    # for key in ["gender", "age", "occupation", "zip"]:
    #     train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values
    
    return train_model_input, train_label



#######
# model
#
#######

class  MatchYoutube(object):
    def __init__(self,*,use_dice=False,is_training=True):
        self.graph = tf.Graph()
        self.tensor_info = {}
        self.depth = dnn_layers
        self.layers = dnn_nums
        self.classes = item_count
        self.dims = emb_dims
        self.is_training = is_training
        self.num_sampled = nums_sampled
        
        self.init_graph()
        self.build_model(self.input)

    def init_graph(self):
        with self.graph.as_default():
            #Main Inputs
            with tf.name_scope('Main_Inputs'):

                self.label_ph = tf.placeholder(tf.float32,[None,],name='target_ph')
                self.lr_ph = tf.placeholder(tf.float32,[],name='lr_ph')
                
                self.user_id_ph = tf.placeholder(tf.int32,[None,],name="user_id")
                self.item_id_ph = tf.placeholder(tf.int32,[None,],name="item_id")
                self.category_id_ph = tf.placeholder(tf.int32,[None,],name="category_id")
                self.tags_id_ph = tf.placeholder(tf.int32,[None,],name='tags_id')
                
                self.event_time_ph = tf.placeholder(tf.float32,[None,],name='event_time')

                self.his_item_seq_ph = tf.placeholder(tf.int32,[None,None],name="hist_item_id")
                self.his_cate_seq_ph = tf.placeholder(tf.int32,[None,None],name="hist_cate_id")
                self.his_tag_seq_ph = tf.placeholder(tf.int32,[None,None],name="hist_tag_id")
                
                self.his_len_ph = tf.placeholder(tf.int32,[None,],name="his_len")

                #title  tfidf w2v  词向量的索引文件,   item的索引文件 
                #description keyword  
                
            with tf.name_scope("Main_Embedding_layer"):
                
                self.user_id_embeddings_var = tf.get_variable("user_id_embeddings_var",
                                                    [user_count,emb_dims],tf.float32,initializer=tf.variance_scaling_initializer())
                self.user_id_embed  = tf.nn.embedding_lookup(self.user_id_embeddings_var,self.user_id_ph)
                
                self.item_id_embeddings_var = tf.get_variable("item_id_embeddings_var",[item_count,emb_dims],initializer=tf.variance_scaling_initializer())
                self.item_id_embed  = tf.nn.embedding_lookup(self.item_id_embeddings_var,self.item_id_ph)
                self.his_item_id_embed  = tf.nn.embedding_lookup(self.item_id_embeddings_var,self.his_item_seq_ph)


 
                self.category_id_embeddings_var = tf.get_variable("category_id_embeddings_var",[category_count,emb_dims],initializer=tf.variance_scaling_initializer())
                self.category_id_embed  = tf.nn.embedding_lookup(self.category_id_embeddings_var,self.category_id_ph)
                self.his_category_id_embed  = tf.nn.embedding_lookup(self.category_id_embeddings_var,self.his_cate_seq_ph)

                self.tags_id_embeddings_var = tf.get_variable("tags_id_embeddings_var",[category_count,emb_dims])
                self.tags_id_embed  = tf.nn.embedding_lookup(self.tags_id_embeddings_var,self.tags_id_ph)
                self.his_tags_id_embed  = tf.nn.embedding_lookup(self.category_id_embeddings_var,self.his_tag_seq_ph)
                
                # seqs


                self.item_his_embed_m  = tf.concat(self.his_item_id_embed+self.his_category_id_embed+self.his_tags_id_embed 
                                      ,axis=-1)
                self.item_his_embed  = tf.reduce_mean(self.item_his_embed_m ,axis=-2, keep_dims=False)

                # with tf.name_scope("nlp"):
                #     self.

            
            self.input = tf.concat([self.user_id_embed,
                self.item_his_embed
                                
                                
                               # tf.reshape(self.category_id_embed,shape=[-1,1]),
                               # tf.reshape(self.tags_id_embed,shape=[-1,1]),
                                
                                ],axis=-1)


    def build_model(self,inp):
        with self.graph.as_default():

            self.saver = tf.train.Saver(max_to_keep=1)

            with tf.name_scope("model"):

                for i in range (self.depth):
                    dnn = tf.layers.dense(inputs = inp,units = self.layers[i],activation=tf.nn.leaky_relu
                    ,kernel_regularizer=tf.contrib.layers.l1_regularizer (0.001),name= 'fc{}'.format(i) )
                    inp = tf.layers.batch_normalization(inputs =dnn ,name = 'fc{}'.format(i)) 
                
                self.output =inp

                weights = tf.get_variable('soft_weight',shape=[self.classes,self.dims],
                      initializer = tf.variance_scaling_initializer())
                biase = tf.get_variable('soft_bias',shape=[self.classes],
                        initializer = tf.variance_scaling_initializer())
                self.items_Mat = weights
                
                self.logits_out = tf.matmul(self.output,tf.transpose(weights))
                
                labels = tf.reshape(self.label_ph,shape=[-1,1])
                with tf.name_scope("loss"):
                    self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                        weights = weights,
                        biases = biase,
                        labels = labels,
                        inputs = self.output,
                        num_sampled=self.num_sampled,
                        num_true=1,
                        num_classes = self.classes
                    ))
                    tf.summary.scalar("loss",self.loss)

                self.optimaizer = tf.train.AdagradOptimizer(lr).minimize(self.loss)
                #self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
    def train(self,sess,inps,merge):
        feed_dict = {}
        for k,v in inps.items():
            feed_dict[getattr(self,k)]=v
    
        loss,_ ,rr = sess.run(
            [self.loss,self.optimaizer,merge],
            feed_dict = feed_dict,
        )

        return loss ,rr

    def test (self,sess,inps):
        feed_dict = {}
        for k,v in inps.items():
            feed_dict[getattr(self,k)]=v

        logits_out,uout,weights = sess.run(
            [self.logits_out,self.output,self.items_Mat],
            feed_dict = feed_dict
        )

        return logits_out,uout,weights

    def cal_acc(self, session, label, logit):
        labels = tf.one_hot(label, self.classes, axis=1)
        correct_pred = tf.equal(tf.argmax(logit, 1), tf.argmax(labels, 1))
        accuracy, labels, logits = session.run([tf.reduce_mean(tf.cast(correct_pred, tf.float32)), labels, logit])
        
        return accuracy, labels, logits

    def save(self,sess,path):
        tf.train.Saver().save(sess,path)
    def restore(self,sess,path):
        tf.train.Saver().restore(sess,path)




def annSearch(emds,uuid,uvec,test_label,ivec,iid,topk):
    
    indextree = faiss.IndexFlatIP(emds)

    faiss.normalize_L2(ivec)
    indextree.add(ivec)

    faiss.normalize_L2(uvec)

    # xq is the query data
# k is the num of neigbors you want to search
# D is the distance matrix between xq and k neigbors
# I is the index matrix of k neigbors
    #D,I = indextree.search(np.ascontiguousarray(uvec),topk)
    D,I = indextree.search(uvec,topk)
    score =[]
    hit =0
    #uid测试集中　用户，，iid所有数据集上的ｉｔｅｍ
    for i , uid in tqdm(enumerate(uuid)):
        try: #每个用户 索引10个结果
            pred = [iid.values[x] for x in I[i]]

            filter_item = None
            #预测的正确的 除以 所有都正样本的  tp/t,,,  tp/p    top10的召回率  每个用户的 然后平均
            recall_score = len(set(pred[:topk]) & set(test_label) ) *1.0/len(test_label)
            score.append(recall_score)
            
            #预测的10个 包括真值, 命中+1
            if test_label[uid] in pred:
                hit+=1
        except:
            print(i)
    score_mean = np.mean(score)

    #每个用户 预测 top10 ,标签命中,  然后除以所有用户,
    hit_rate = hit /len(uuid) #uuid.shape[0] #

    return score_mean,hit_rate
    





if __name__ == '__main__':

    #dataframe = get_data_from_csv('/home/tv/Downloads/query_result_desc.csv')
    #df_new =compute_index_tags(dataframe)

    df_new = pd.read_csv('/home/tv/Downloads/query_result_desc_new.csv',lineterminator="\n")

    model = MatchYoutube(is_training=True)

    with tf.Session(graph=model.graph) as sess:
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("./logs",sess.graph)
        summary_writer.add_graph(sess.graph)#添加graph图



        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())




        
        n_batches = int(np.ceil(len(df_new))/batch_size)

        try:
            for i in range(epoches):

                for j in range(n_batches):

                    train_set,test_set = gen_data_set(df_new)

                    next_batch = train_set
                    
                    #train_set,test_set = gen_data_set(next_batch)   

                    traindata,train_label= gen_model_input(next_batch,item_his_max_len)
                    
                    traindata["label_ph"] = train_label
                    model.is_training = True
                    loss,rs = model.train(sess,traindata,merged)

                    print("epoch: {},batch: {},loss {}".format(i,j,loss))

                    summary_writer.add_summary(rs,i*epoches+j)
                    if (j)%1 ==0 :
                        model.save(sess, './model/tt')

                    if  (j) %5 ==0:
                        
                        test_uid = [line[0] for line in test_set]
                        test_true_label = {line[0]:[line[2]] for line in test_set}
                         

                        testdata,test_label= gen_model_input(test_set,item_his_max_len)
                        model.is_training = False
                        model.restore(sess,'./model/tt')
                        logits_out,user_out,weights = model.test(sess,testdata) 

                        np.save('./weights.npy' , weights)
                        
                        for k in [5,10,20,30]:
                            score_mean,hit_rate = annSearch(emb_dims,test_uid,user_out,
                        
                            test_true_label, weights, df_new['item_id'],k)
                            print("recall: {}, hit_rate: {}, topk: {}".format(score_mean,hit_rate,k))

        except Exception as e:
            print(e)
            logging.debug("{} epoch: ".format(i))
            logging.debug("execept e: {}".format(traceback.format_exc()))





            24599it [00:14, 1647.47it/s]
recall: 0.00013739962972500304, hit_rate: 0.0048375950241879755, topk: 5
24599it [00:15, 1547.65it/s]
recall: 0.00027148912187406154, hit_rate: 0.010650839465018902, topk: 10
24599it [00:17, 1378.27it/s]
recall: 0.000533237878859148, hit_rate: 0.022927761291109395, topk: 20
24599it [00:20, 1215.90it/s]
recall: 0.0007748250390261998, hit_rate: 0.035692507825521365, topk: 30