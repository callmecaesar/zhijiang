import json
import numpy as np
import pandas as pd
import jieba

class GetData:
    def __init__(self, granu):
        self.data_save_path_root = '../data/'
        self.chunk_tags = ['O', 'B_AT', 'I_AT', 'B_OT', 'I_OT']
        self.categories = ['价格', '使用体验', '其他', '功效', '包装', '尺寸', '成分', '整体', '新鲜度', '服务', '气味', '物流', '真伪']
        self.chunk2id, self.id2chunk, self.categories2id, self.id2categories = self._reshape_data('all')
        self.polarities2id = {'正面': 1, '负面': 0}
        self.id2polarities = {1: '正面', 0: '负面'}
        fr = open('../data/aspect_opinion_' + granu + '_data.json', 'r')
        self.polarities_categories_train_data = json.load(fr)
        self.granu = granu
    def _reshape_data(self, mode):
        """
            mode ->  'all' : 标注在一起
                     ->  'at' : AspectTerm only
                     ->  'ot' : OpinionTerms only
        """
        if mode == 'all':
            chunk_tags = self.chunk_tags
        elif mode == 'at':
            chunk_tags = ['O', 'B_AT', 'I_AT']
        elif mode == 'ot':
            chunk_tags = ['O', 'B_OT', 'I_OT']
            
        chunk2id = {item: _id for _id, item in enumerate(chunk_tags)}
        id2chunk = {v: k for k, v in chunk2id.items()}
        categories2id = {item: _id for _id, item in enumerate(self.categories)}
        id2categories = {v: k for k, v in categories2id.items()}
        return chunk2id, id2chunk, categories2id, id2categories
    
    def get_ner_train_data(self):
        with open(self.data_save_path_root + self.granu + '_level.json', 'r', encoding='utf8') as fr:
            vocab_dict = json.load(fr)
        with open(self.data_save_path_root + 'ner_' + self.granu + '_data.json', 'r', encoding='utf8') as fr:
            ner_data = json.load(fr)
        with open(self.data_save_path_root + 'ner_' + self.granu + '_label.json', 'r', encoding='utf8') as fr:
            ner_label = json.load(fr)
            
            
        vocab2id = vocab_dict.get(self.granu + '2id')
        id2vocab = vocab_dict.get('id2' + self.granu)
        
        for i in range(len(ner_data)):
            ner_data[i] = np.asarray([vocab2id.get(item, 1) for item in ner_data[i]])
            
        ner_data = np.asarray(ner_data)
        
        for i in range(len(ner_label)):
            ner_label[i] = np.asarray([self.chunk2id.get(item, 1) for item in ner_label[i]])
            
        ner_data = np.asarray(ner_data)
        ner_label = np.asarray(ner_label)
        
        
        return self.chunk2id, self.id2chunk, vocab2id, id2vocab, ner_data, ner_label
    
    def get_categories_train_data(self):
        return self.polarities_categories_train_data.get('content'), self.polarities_categories_train_data.get('aspect'), self.polarities_categories_train_data.get('opinion'), self.polarities_categories_train_data.get('category'), self.polarities_categories_train_data.get('id2categories')
    def get_polarities_train_data(self):
        return self.polarities_categories_train_data.get('content'), self.polarities_categories_train_data.get('aspect'), self.polarities_categories_train_data.get('opinion'), self.polarities_categories_train_data.get('polarity'), self.id2polarities


    def get_test_data(self, test_file_path, vocab2id):
        test_id = []
        test_data = []
        table = pd.read_csv(test_file_path)
        for index, row in table.iterrows():
            _id = row['id']
            content = row['Reviews']
            if self.granu == 'char':
                words = list(content)
            else:
                words = jieba.lcut(content)
            test_id.append(_id)
            test_data.append([vocab2id.get(w, 1) for w in words])
            
        return test_id, test_data
