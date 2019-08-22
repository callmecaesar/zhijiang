import jieba
from collections import Counter
import json
import pandas as pd
train_reviews_path = '../data/Train_reviews.csv'
train_labels_path = '../data/Train_labels.csv'
class ProcessData:
    def __init__(self, train_reviews_path, train_labels_path):
        self.data_save_path_root = '../data/'
        print('数据预处理...')
        train_reviews = pd.read_csv(train_reviews_path)
        train_labels = pd.read_csv(train_labels_path)
        self.table = train_reviews.join(train_labels.set_index('id'), how='right', on='id')
        self.groups = self.table.groupby('id')
        print('数据表整合完成...')
        self.chunk_tags = ['B_AT', 'I_AT', 'O', 'B_OT', 'I_OT']
        self.Categories2id, self.id2Categories = self._get_Categories()
        
    def _reshape_data(self, mode, cut):
        """
            mode ->  'all' : 标注在一起
                     ->  'at' : AspectTerm only
                     ->  'ot' : OpinionTerms only
        """
        if mode == 'all':
            chunk_tags = self.chunk_tags
        elif mode == 'at':
            chunk_tags = self.chunk_tags[:3]
        elif mode == 'ot':
            chunk_tags = self.chunk_tags[2:]
            
        self.chunk2id = {item: _id for _id, item in enumerate(chunk_tags)}
        self.id2chunk = {v: k for k, v in self.chunk2id}
        
    def _get_Categories(self):
        Categories = set(list(self.table['Categories']))
        
        Categories2id = {item: _id for _id, item in enumerate(Categories)}
        id2Categories = {v: k for k, v in Categories2id.items()}
        
        return Categories2id, id2Categories
        
    def _save_data(self, save_file, cut):
        ner_data = []
        ner_label = []
#         category = []
#         polarity = []
        with open(save_file, 'w') as f:
            for index, g in self.groups:
                Reviews = list(g.Reviews)
                AspectTerms = list(g.AspectTerms)
                OpinionTerms = list(g.OpinionTerms)
                Categories = list(g.Categories)
                Polarities = list(g.Polarities)
                A_start = list(g.A_start)
                A_end = list(g.A_end)
                O_start = list(g.O_start)
                O_end = list(g.O_end)

                sentence = Reviews[0].strip()
                if cut:
                    
                    sentence = jieba.lcut(sentence)
                    temp_chunk_list = []
                    temp_dic = {}
                   
                    for i in range(len(AspectTerms)):
                        if A_start[i] != ' ':
                            temp_dic[int(A_start[i])] = 'A' + AspectTerms[i]
                    for i in range(len(OpinionTerms)):
                        if O_start[i] != ' ':
                            temp_dic[int(O_start[i])] = 'O' + OpinionTerms[i]
                    
                    pairs = sorted(temp_dic.items(), key = lambda x:x[0])
                    
                    flag = 0
                    for item in pairs:
                        key = item[1]
                        AO = key[0]
                        key = key[1:]
                        while(flag < len(sentence)):
                            if sentence[flag] in key:
                                
                                if key.index(sentence[flag]) == 0:
                                    if AO == 'A':
                                        temp_chunk_list.append('B_AT')
                                    else:
                                        temp_chunk_list.append('B_OT')
                                    
                                    if sentence[flag][-1] in key and key.index(sentence[flag][-1]) + 1 == len(key):
                                        flag += 1
                                        break
                                    flag += 1
                                else:
                                    if AO == 'A':
                                        temp_chunk_list.append('I_AT')
                                    else:
                                        temp_chunk_list.append('I_OT')
                                    if sentence[flag][-1] in key and key.index(sentence[flag][-1]) + 1 == len(key):
                                        flag += 1
                                        break
                                    flag += 1
                            else:
                                temp_chunk_list.append('O')
                                flag += 1
                    temp_chunk_list = temp_chunk_list + ['O'] * (len(sentence) - len(temp_chunk_list))       
                    for i in range(len(sentence)):
                        f.write(sentence[i] + '\t' +temp_chunk_list[i] + '\n')
                    f.write('\n')
                else:
                    sentence = list(sentence)
                    temp_chunk_list = ['O'] * len(sentence)
                    for i in range(len(A_start)):
                        if A_start[i] != ' ':
                            b = int(A_start[i])
                            e = int(A_end[i])
                            temp_chunk_list[b] = 'B_AT'
                            temp_chunk_list[b+1: e] = ["I_AT"]*(e-b-1)

                        if O_start[i] != ' ':
                            b = int(O_start[i])
                            e = int(O_end[i])
                            temp_chunk_list[b] = 'B_OT'
                            temp_chunk_list[b+1: e] = ["I_OT"]*(e-b-1)

                    for i in range(len(sentence)):
                        f.write(sentence[i] + '\t' +temp_chunk_list[i] + '\n')

                    f.write('\n')
                    
                ner_data.append(sentence)
                ner_label.append(temp_chunk_list)
#                 category.append()
#                 polarity.append()
        ner_data = json.dumps(ner_data, ensure_ascii=False)
        ner_label = json.dumps(ner_label, ensure_ascii=False)
        
        return ner_data, ner_label
            
            
    def _save_vcab(self, thred=1):
        word_counts = {}
        char_counts = {}
        for index, g in self.groups:
            Review = list(g.Reviews)[0]
            words = jieba.lcut(Review)
            
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1
                
            for c in Review:
                char_counts[c] = char_counts.get(c, 0) + 1
                
        word_counts = {i: j for i, j in word_counts.items() if j > thred}
        word2id = { word: _id+2 for _id, word in enumerate(word_counts)}
        char2id = { char: _id+2 for _id, char in enumerate(char_counts)}
        word2id['UNK'] = 1
        word2id['PAD'] = 0
        char2id['UNK'] = 1
        char2id['PAD'] = 0
        
        id2word = {v: k for k, v in word2id.items()}
        id2char = {v: k for k, v in char2id.items()}
        
        return word2id, id2word, word_counts, char2id, id2char, char_counts
    
    def sava_data(self):
        
        word2id, id2word, word_counts, char2id, id2char, char_counts = self._save_vcab()
        
        word_level = {}
        
        print('字典生成与存储...')
        
        word_level['word2id'] = word2id
        word_level['id2word'] = id2word
        word_level['word_counts'] = word_counts
        
        json_data = json.dumps(word_level, ensure_ascii=False)
        
        with open(self.data_save_path_root + 'word_level.json', 'w', encoding='utf8') as fw:
            fw.write(json_data)
            
        char_level = {}
        char_level['char2id'] = char2id
        char_level['id2char'] = id2char
        char_level['char_counts'] = char_counts
        
        json_data = json.dumps(char_level, ensure_ascii=False)
        
        print('字典生成与存储完毕')
        print('数据生成与存储...')
        with open(self.data_save_path_root + 'char_level.json', 'w', encoding='utf8') as fw:
            fw.write(json_data)
       
        ner_word_data, ner_word_label = self._save_data(self.data_save_path_root + 'word_train.txt', True)
        ner_char_data, ner_char_label = self._save_data(self.data_save_path_root + 'char_train.txt', False)
        with open(self.data_save_path_root + 'ner_word_data.json', 'w', encoding='utf8') as fw:
            fw.write(ner_word_data)
        with open(self.data_save_path_root + 'ner_word_label.json', 'w', encoding='utf8') as fw:
            fw.write(ner_word_label)
        with open(self.data_save_path_root + 'ner_char_data.json', 'w', encoding='utf8') as fw:
            fw.write(ner_char_data)
        with open(self.data_save_path_root + 'ner_char_label.json', 'w', encoding='utf8') as fw:
            fw.write(ner_char_label)
            
        self._get_category_polarity(char2id, word2id, self.Categories2id)
        print('数据生成与存储完毕！')
        
    def _get_category_polarity(self, char2id, word2id, Categories2id):
        #Reviews	AspectTerms	A_start	A_end	OpinionTerms	O_start	O_end	Categories	Polarities
        
        content_word = []
        aspect_word = []
        opinion_word = []
        
        content_char = []
        aspect_char = []
        opinion_char = []
        
        category = []
        polarity = []
        
        for index, row in self.table.iterrows():
            Reviews = row['Reviews']
            AspectTerms = row['AspectTerms']
            OpinionTerms = row['OpinionTerms']
            Categories = row['Categories']
            Polarities = row['Polarities']
            
            content_word.append([word2id.get(w) for w in jieba.lcut(Reviews)])
            content_char.append([char2id.get(w) for w in Reviews])
            if AspectTerms == '_':
                aspect_word.append([0])
                aspect_char.append([0])
            else:
                aspect_word.append([word2id.get(w) for w in jieba.lcut(AspectTerms)])
                aspect_char.append([char2id.get(w) for w in AspectTerms])
                
            if OpinionTerms == '_':
                opinion_word.append([0])
                opinion_char.append([0])
            else:
                opinion_word.append([word2id.get(w) for w in jieba.lcut(OpinionTerms)])
                opinion_char.append([char2id.get(w) for w in OpinionTerms])
            category.append(Categories2id.get(Categories))
            if Polarities == '正面':
                polarity.append(1)
            else:
                polarity.append(0)
                
        word_dict = {'content': content_word, 'aspect': aspect_word, 'opinion': opinion_word, 'polarity': polarity, 'category': category, 'id2categories':self.id2Categories}
        char_dict = {'content': content_char, 'aspect': aspect_char, 'opinion': opinion_char, 'polarity': polarity, 'category': category, 'id2categories':self.id2Categories}
        
        word_dict = json.dumps(word_dict)
        char_dict = json.dumps(char_dict)
        with open(self.data_save_path_root + 'aspect_opinion_word_data.json', 'w', encoding='utf8') as fw:
            fw.write(word_dict)
        with open(self.data_save_path_root + 'aspect_opinion_char_data.json', 'w', encoding='utf8') as fw:
            fw.write(char_dict)