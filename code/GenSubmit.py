import jieba.posseg as psg
from copy import copy
import pandas as pd
class GenSubmit:
    def __init__(self, corpus, id2word):
        self.id2word = id2word
        self.punct_set = self._get_punct(corpus)
        
    def _get_punct(self, corpus):
        punct_set = []
        for item in corpus:
            pos = psg.lcut(item)
            for pos_pair in pos:
                if list(pos_pair)[1] == 'x':
                    punct_set.append(list(pos_pair)[0])
        punct_set = list(set(punct_set))
        punct_set = [item for item in punct_set if not item.isdigit()]
        return punct_set
    
    def _get_entity_list(self, words_list, chunk_list):
        candidate_a = []
        candidate_o = []
        a_list = []
        o_list = []
        words_list = [str(item) for item in words_list]
        for i in range(len(words_list)):
            if chunk_list[i] == 'B_AT':
                if len(candidate_a) != 0:
                    a_list.append(''.join(candidate_a))
                    candidate_a = []
                if len(candidate_o) != 0:
                    o_list.append(''.join(candidate_o))
                    candidate_o = []
                candidate_a.append(self.id2word.get(words_list[i]))
                
            elif chunk_list[i] == 'I_AT':
                candidate_a.append(self.id2word.get(words_list[i]))
                
            elif chunk_list[i] == 'B_OT':
                if len(candidate_o) != 0:
                    o_list.append(''.join(candidate_o))
                    candidate_o = []
                if len(candidate_a) != 0:
                    a_list.append(''.join(candidate_a))
                    candidate_a = []
                candidate_o.append(self.id2word.get(words_list[i]))
                
            elif chunk_list[i] == 'I_OT':
                candidate_o.append(self.id2word.get(words_list[i]))
                
            else:
                if len(candidate_a) != 0:
                    a_list.append(''.join(candidate_a))

                if len(candidate_o) != 0:
                    o_list.append(''.join(candidate_o))
                    
                candidate_a = []
                candidate_o = []
        if len(candidate_o) != 0:
            o_list.append(''.join(candidate_o))
        if len(candidate_a) != 0:
            a_list.append(''.join(candidate_a))
        return a_list, o_list
    
    def _get_processed_sent(self, sent):
        new_sent = sent
        for punct in self.punct_set:
            new_sent = new_sent.replace(punct, '$$')
            
        new_sent = new_sent.strip('$$')
        return new_sent
    
    def _detect_relation(self, sent_parts, aspect, option):
        for sub_sent in sent_parts:
            if aspect in sub_sent and option in sub_sent:
                return True
            
        return False
        
    def get_ner_result(self, id_list, sent_list, words_list, chunk_list):
        result_id = []
        result_Reviews = []
        result_AspectTerms = []
        result_A_start = []
        result_A_end = []
        result_OpinionTerms = []
        result_O_start = []
        result_O_end = []
        for i in range(len(sent_list)):
            a_list, o_list = self._get_entity_list(words_list[i], chunk_list[i])
            if len(a_list) == 0 and len(o_list) == 0:
                result_id.append(id_list[i])
                result_Reviews.append(sent_list[i])
                result_AspectTerms.append('_')
                result_OpinionTerms.append('_')
                result_A_start.append(' ')
                result_A_end.append(' ')
                result_O_start.append(' ')
                result_O_end.append(' ')
                
            elif len(a_list) == 0:
                for o in o_list:
                    o = str(o)
                    if o not in sent_list[i]:
                        continue
                    result_id.append(id_list[i])
                    result_Reviews.append(sent_list[i])
                    result_AspectTerms.append('_')
                    result_OpinionTerms.append(o)
                    result_A_start.append(' ')
                    result_A_end.append(' ')
                    result_O_start.append(sent_list[i].index(o))
                    result_O_end.append(sent_list[i].index(o) + len(o))
                    
            elif len(o_list) == 0:
                for a in a_list:
                    a = str(a)
                    if a not in sent_list[i]:
                        continue
                    result_id.append(id_list[i])
                    result_Reviews.append(sent_list[i])
                    result_AspectTerms.append(a)
                    result_OpinionTerms.append('_')
                    result_A_start.append(sent_list[i].index(a))
                    result_A_end.append(sent_list[i].index(a) + len(a))
                    result_O_start.append(' ')
                    result_O_end.append(' ')
            else:
                new_sent = self._get_processed_sent(sent_list[i])
                sent_parts = new_sent.split('$$')
                recorded_a = []
                recorded_o = []
                for ai in range(len(a_list)):
                    a = a_list[ai]
                    a = str(a)
                    if a not in sent_list[i]:
                        continue
                    for oi in range(len(o_list)):
                        o = o_list[oi]
                        o = str(o)
                        if o not in sent_list[i]:
                            continue
                        if self._detect_relation(sent_parts, a, o):
                            result_id.append(id_list[i])
                            result_Reviews.append(sent_list[i])
                            result_AspectTerms.append(a)
                            result_OpinionTerms.append(o)
                            result_A_start.append(sent_list[i].index(a))
                            result_A_end.append(sent_list[i].index(a) + len(a))
                            result_O_start.append(sent_list[i].index(o))
                            result_O_end.append(sent_list[i].index(o) + len(o))
                            
                            recorded_a.append(ai)
                            recorded_o.append(oi)
                for ai in range(len(a_list)):
                    if ai not in recorded_a:
                        a = a_list[ai]
                        a = str(a)
                        if a not in sent_list[i]:
                            continue
                        result_id.append(id_list[i])
                        result_Reviews.append(sent_list[i])
                        result_AspectTerms.append(a)
                        result_OpinionTerms.append('_')
                        result_A_start.append(sent_list[i].index(a))
                        
                        result_A_end.append(sent_list[i].index(a) + len(a))
                        result_O_start.append(' ')
                        result_O_end.append(' ')
                        
                for oi in range(len(o_list)):
                    if oi not in recorded_o:
                        o = o_list[oi]
                        o = str(o)
                        if o not in sent_list[i]:
                            continue
                        result_id.append(id_list[i])
                        result_Reviews.append(sent_list[i])
                        result_AspectTerms.append('_')
                        result_OpinionTerms.append(o)
                        result_A_start.append(' ')
                        result_A_end.append(' ')
                        # print(o)
                        # print(sent_list[i])
                        result_O_start.append(sent_list[i].index(o))
                        result_O_end.append(sent_list[i].index(o) + len(o))
                        
        ret_table = pd.DataFrame()
        ret_table['id'] = result_id
        ret_table['Reviews'] = result_Reviews
        ret_table['AspectTerms'] = result_AspectTerms
        ret_table['A_start'] = result_A_start
        ret_table['A_end'] = result_A_end
        ret_table['OpinionTerms'] = result_OpinionTerms
        ret_table['O_start'] = result_O_start
        ret_table['O_end'] = result_O_end
        
        return ret_table