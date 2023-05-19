import json
import os
import numpy as np
import random
import torch
from torch.autograd import Variable

from transformers import BertTokenizer, BertModel, BertForMaskedLM

class JSONFileDataLoader:
    def _load_preprocessed_file(self):
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        self.uid = np.load('./data/' + name_prefix + '_uid.npy')
        word_vec_name_prefix = '.'.join(self.word_vec_file_name.split('/')[-1].split('.')[:-1])
        processed_data_dir = '_processed_data'
        if not os.path.isdir(processed_data_dir):
            return False
        word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        pos1_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos1.npy')
        pos2_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos2.npy')
        mask_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask.npy')
        length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        label_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_label.npy')
        entpair_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_entpair.npy')
        rel2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2scope.json')
        word_vec_mat_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy')
        word2id_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json')
        rel2id_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2id.json')
        entpair2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_entpair2scope.json')
        if not os.path.exists(word_npy_file_name) or \
                not os.path.exists(pos1_npy_file_name) or \
                not os.path.exists(pos2_npy_file_name) or \
                not os.path.exists(mask_npy_file_name) or \
                not os.path.exists(length_npy_file_name) or \
                not os.path.exists(label_npy_file_name) or \
                not os.path.exists(entpair_npy_file_name) or \
                not os.path.exists(rel2scope_file_name) or \
                not os.path.exists(word_vec_mat_file_name) or \
                not os.path.exists(word2id_file_name) or \
                not os.path.exists(rel2id_file_name) or \
                not os.path.exists(entpair2scope_file_name):
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name)
        self.data_pos1 = np.load(pos1_npy_file_name)
        self.data_pos2 = np.load(pos2_npy_file_name)
        self.data_mask = np.load(mask_npy_file_name)
        self.data_length = np.load(length_npy_file_name)
        self.data_label = np.load(label_npy_file_name)
        self.data_entpair = np.load(entpair_npy_file_name)
        self.rel2scope = json.load(open(rel2scope_file_name))
        self.rel2id = json.load(open(rel2id_file_name))
        self.word_vec_mat = np.load(word_vec_mat_file_name)
        self.word2id = json.load(open(word2id_file_name))
        self.rel_tot = len(self.rel2id)
        if self.data_word.shape[1] != self.max_length:
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def __init__(self, file_name, word_vec_file_name, max_length=40, case_sensitive=False, reprocess=False, distant=False):
        '''
        file_name: Json file storing the data in the following format
            {
                "P155": # relation id
                    [
                        {
                            "h": ["song for a future generation", "Q7561099", [[16, 17, ...]]], # head entity [word, id, location]
                            "t": ["whammy kiss", "Q7990594", [[11, 12]]], # tail entity [word, id, location]
                            "tokens": ["Hot", "Dance", "Club", ...], # sentence
                        },
                        ...
                    ],
                "P177": 
                    [
                        ...
                    ]
                ...
            }
        word_vec_file_name: Json file storing word vectors in the following format
            [
                {'word': 'the', 'vec': [0.418, 0.24968, ...]},
                {'word': ',', 'vec': [0.013441, 0.23682, ...]},
                ...
            ]
        max_length: The length that all the sentences need to be extend to.
        case_sensitive: Whether the data processing is case-sensitive, default as False.
        reprocess: Do the pre-processing whether there exist pre-processed files, default as False.
        '''
        self.file_name = file_name
        self.word_vec_file_name = word_vec_file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        self.current = 0

        if reprocess or not self._load_preprocessed_file():  # Try to load pre-processed files:
            # Check files
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")
            if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
                raise Exception("[ERROR] Word vector file doesn't exist")

            # Load files
            print("Loading data file...")
            self.ori_data = json.load(open(self.file_name, "r"))
            print("Finish loading")
            print("Loading word vector file...")
            self.ori_word_vec = json.load(open(self.word_vec_file_name, "r"))
            print("Finish loading")

            # Eliminate case sensitive
            if not case_sensitive:
                print("Elimiating case sensitive problem...")
                for relation in self.ori_data:
                    for ins in self.ori_data[relation]:
                        for i in range(len(ins['tokens'])):
                            ins['tokens'][i] = ins['tokens'][i].lower()
                print("Finish eliminating")

            # Pre-process word vec
            self.word2id = {}
            self.word_vec_tot = len(self.ori_word_vec)
            UNK = self.word_vec_tot
            BLANK = self.word_vec_tot + 1
            self.word_vec_dim = len(self.ori_word_vec[0]['vec'])
            print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
            print("Building word vector matrix and mapping...")
            self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
            for cur_id, word in enumerate(self.ori_word_vec):
                w = word['word']
                if not case_sensitive:
                    w = w.lower()
                self.word2id[w] = cur_id
                self.word_vec_mat[cur_id, :] = word['vec']
                self.word_vec_mat[cur_id] = self.word_vec_mat[cur_id] / np.sqrt(np.sum(self.word_vec_mat[cur_id] ** 2))
            self.word2id['UNK'] = UNK
            self.word2id['BLANK'] = BLANK
            print("Finish building")

            # Pre-process data
            print("Pre-processing data...")
            self.instance_tot = 0
            for relation in self.ori_data:
                self.instance_tot += len(self.ori_data[relation])
            self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos1 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos2 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_mask = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
            self.data_label = np.zeros((self.instance_tot), dtype=np.int32)
            self.data_entpair = []
            self.rel2scope = {}  # left closed and right open
            self.entpair2scope = {}

            self.rel2id = {}
            self.rel_tot = 0
            #TODO
            tokenizer = BertTokenizer.from_pretrained('D:/2022_2_GraphTransformer\graphtransformer/Neural-Snowball_\data/bert-base-uncased')

            i = 0
            for relation in self.ori_data:
                self.rel2scope[relation] = [i, i]
                if relation not in self.rel2id:
                    self.rel2id[relation] = self.rel_tot
                    self.rel_tot += 1

                for ins in self.ori_data[relation]:
                    if distant:
                        head = ins['h']['name']
                        tail = ins['t']['name']
                        pos1 = ins['h']['pos'][0][0]
                        pos2 = ins['t']['pos'][0][0]
                        pos1_end = ins['h']['pos'][0][-1]
                        pos2_end = ins['t']['pos'][0][-1]
                    else:
                        head = ins['h'][0]
                        tail = ins['t'][0]
                        pos1 = ins['h'][2][0][0]
                        pos2 = ins['t'][2][0][0]
                        pos1_end = ins['h'][2][0][-1]
                        pos2_end = ins['t'][2][0][-1]
                    words = ins['tokens']
                    entpair = head + '#' + tail
                    self.data_entpair.append(entpair)

                    ## TODO
                    # tokenize
                    # # head entity # @ tail entity @
                    if pos1 < pos2:
                        new_words = ['[CLS]'] + words[:pos1] + ['#'] + words[pos1:pos1_end + 1] + ['#'] + words[
                                                                                                          pos1_end + 1:pos2] \
                                    + ['@'] + words[pos2:pos2_end + 1] + ['@'] + words[pos2_end + 1:]
                    else:
                        new_words = ['[CLS]'] + words[:pos2] + ['@'] + words[pos2:pos2_end + 1] + ['@'] + words[
                                                                                                          pos2_end + 1:pos1] \
                                    + ['#'] + words[pos1:pos1_end + 1] + ['#'] + words[pos1_end + 1:]
                    sentence = ' '.join(new_words)
                    '''
                    tmp = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))

                    self.data_length[i] = min(len(tmp), max_length)
                    if len(tmp) < max_length:
                        self.data_mask[i][:len(tmp)] = 1
                        tmp += [0] * (max_length - len(tmp))
                    else:
                        tmp = tmp[:max_length]
                        self.data_mask[i][:] = 1
                        

                    self.data_word[i][:] = np.array(tmp)
                    '''
                    self.data_label[i] = self.rel2id[relation]
                    #if len(words) > max_length:
                    #    self.data_length[i] = max_length
                    if not entpair in self.entpair2scope:
                        self.entpair2scope[entpair] = [i]
                    else:
                        self.entpair2scope[entpair].append(i)
                    #i += 1
                #self.rel2scope[relation][1] = i

                    
                    cur_ref_data_word = self.data_word[i]
                    for j, word in enumerate(words):
                        if j < max_length:
                            if word in self.word2id:
                                cur_ref_data_word[j] = self.word2id[word]
                            else:
                                cur_ref_data_word[j] = UNK
                        else:
                            break
                    for j in range(j + 1, max_length):
                        cur_ref_data_word[j] = BLANK
                    self.data_length[i] = len(words)
                    if len(words) > max_length:
                        self.data_length[i] = max_length
                    if pos1 >= max_length:
                        pos1 = max_length - 1
                    if pos2 >= max_length:
                        pos2 = max_length - 1
                    pos_min = min(pos1, pos2)
                    pos_max = max(pos1, pos2)
                    for j in range(max_length):
                        self.data_pos1[i][j] = j - pos1 + max_length
                        self.data_pos2[i][j] = j - pos2 + max_length
                        if j >= self.data_length[i]:
                           # self.data_mask[i][j] = 0
                            self.data_pos1[i][j] = 0
                            self.data_pos2[i][j] = 0
                        elif j <= pos_min:
                            self.data_mask[i][j] = 1
                        elif j <= pos_max:
                            self.data_mask[i][j] = 2
                        else:
                            self.data_mask[i][j] = 3

                    i += 1
                self.rel2scope[relation][1] = i



            print("Finish pre-processing")
            print("Storing processed files...")
            name_prefix = '.'.join(file_name.split('/')[-1].split('.')[:-1])
            word_vec_name_prefix = '.'.join(word_vec_file_name.split('/')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)
            self.data_entpair = np.array(self.data_entpair)
            np.save(os.path.join(processed_data_dir, name_prefix + '_word.npy'), self.data_word)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos1.npy'), self.data_pos1)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos2.npy'), self.data_pos2)
            np.save(os.path.join(processed_data_dir, name_prefix + '_mask.npy'), self.data_mask)
            np.save(os.path.join(processed_data_dir, name_prefix + '_length.npy'), self.data_length)
            np.save(os.path.join(processed_data_dir, name_prefix + '_label.npy'), self.data_label)
            np.save(os.path.join(processed_data_dir, name_prefix + '_entpair.npy'), self.data_entpair)
            json.dump(self.rel2scope, open(os.path.join(processed_data_dir, name_prefix + '_rel2scope.json'), 'w'))
            json.dump(self.rel2id, open(os.path.join(processed_data_dir, name_prefix + '_rel2id.json'), 'w'))
            np.save(os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy'), self.word_vec_mat)
            json.dump(self.word2id, open(os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json'), 'w'))
            print("Finish storing")
        self.index = list(range(self.instance_tot))
        self.shuffle = True
        if self.shuffle:
            random.shuffle(self.index)


    def next_one(self, N, K, Q): #Q 5 5 100
        target_classes = random.sample(self.rel2scope.keys(), N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': [], 'entpair': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id':[]}
        query_label = []

        for i, class_name in enumerate(target_classes):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), K + Q, False)
            word = self.data_word[indices]
            pos1 = self.data_pos1[indices]
            pos2 = self.data_pos2[indices]
            mask = self.data_mask[indices]
            id = self.uid[indices]
            support_word, query_word = np.split(word, [K])
            support_pos1, query_pos1 = np.split(pos1, [K])
            support_pos2, query_pos2 = np.split(pos2, [K])
            support_mask, query_mask = np.split(mask, [K])
            support_id, query_id, _ = np.split(id, [K, K + Q])
            support_entpair = list(self.data_entpair[indices[:K]])
            support_set['word'].append(support_word)
            support_set['pos1'].append(support_pos1)
            support_set['pos2'].append(support_pos2)
            support_set['mask'].append(support_mask)
            support_set['id'] = np.concatenate([support_set['id'], support_id], axis=0)
            support_set['entpair'] = np.concatenate([support_set['entpair'], support_entpair], axis=0)
            query_set['word'].append(query_word)
            query_set['pos1'].append(query_pos1)
            query_set['pos2'].append(query_pos2)
            query_set['mask'].append(query_mask)
            query_set['id'].append(query_id)
            query_label += [i] * Q

        support_set['word'] = np.stack(support_set['word'], 0)
        support_set['pos1'] = np.stack(support_set['pos1'], 0)
        support_set['pos2'] = np.stack(support_set['pos2'], 0)
        support_set['mask'] = np.stack(support_set['mask'], 0)
        support_set['id'] = np.stack(support_set['id'], 0)
        support_set['entpair'] = np.stack(support_set['entpair'], 0)
        query_set['word'] = np.concatenate(query_set['word'], 0)
        query_set['pos1'] = np.concatenate(query_set['pos1'], 0)
        query_set['pos2'] = np.concatenate(query_set['pos2'], 0)
        query_set['mask'] = np.concatenate(query_set['mask'], 0)
        query_set['id'] = np.concatenate(query_set['id'], 0)
        query_label = np.array(query_label)

        return support_set, query_set, query_label, target_classes

    def next_batch(self, B=4, N=20, K=5, Q=100):
        support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': [], 'entpair': []}
        query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': []}
        label = []
        #support, query, label, target_classes = self.next_one(N, K, Q)

        for one_sample in range(B):
            current_support, current_query, current_label, target_classes = self.next_one(N, K, Q)
            support['word'].append(current_support['word'])
            support['pos1'].append(current_support['pos1'])
            support['pos2'].append(current_support['pos2'])
            support['mask'].append(current_support['mask'])
            support['id'] = np.concatenate([support['id'], current_support['id']], axis=0)
            support['entpair'] = np.concatenate([support['entpair'], current_support['entpair']], axis=0)
            query['word'].append(current_query['word'])
            query['pos1'].append(current_query['pos1'])
            query['pos2'].append(current_query['pos2'])
            query['mask'].append(current_query['mask'])
            query['id'].append(current_query['id'])
            label.append(current_label)

        support['word'] = torch.from_numpy(np.stack(support['word'], 0)).long().view(-1, self.max_length)
        support['pos1'] = torch.from_numpy(np.stack(support['pos1'], 0)).long().view(-1, self.max_length)
        support['pos2'] = torch.from_numpy(np.stack(support['pos2'], 0)).long().view(-1, self.max_length)
        support['mask'] = torch.from_numpy(np.stack(support['mask'], 0)).long().view(-1, self.max_length)
        support['id'] = Variable(torch.from_numpy(np.stack(support['id'], 0)).long())
        support['entpair'] = support['entpair'].tolist()
        query['word'] = torch.from_numpy(np.stack(query['word'], 0)).long().view(-1, self.max_length)
        query['pos1'] = torch.from_numpy(np.stack(query['pos1'], 0)).long().view(-1, self.max_length)
        query['pos2'] = torch.from_numpy(np.stack(query['pos2'], 0)).long().view(-1, self.max_length)
        query['mask'] = torch.from_numpy(np.stack(query['mask'], 0)).long().view(-1, self.max_length)
        query['id'] = Variable(torch.from_numpy(np.stack(query['id'], 0)).long())
        label = torch.from_numpy(np.stack(label, 0).astype(np.int64)).long()

        for key in ['word', 'pos1', 'pos2', 'mask', 'id']:
            support[key] = support[key].cuda()
        #for key in support:
         #   support[key] = support[key].cuda()
        for key in query:
            query[key] = query[key].cuda()
        label = label.cuda()

        return support, query, label, target_classes
    def neg_batch(self, batch_size):
        batch = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': []}
        if self.current + batch_size > len(self.index):
            self.index = list(range(self.instance_tot))
            if self.shuffle:
                random.shuffle(self.index)
            self.current = 0
        current_index = self.index[self.current:self.current+batch_size]
        self.current += batch_size

        batch['word'] = Variable(torch.from_numpy(self.data_word[current_index]).long())
        batch['pos1'] = Variable(torch.from_numpy(self.data_pos1[current_index]).long())
        batch['pos2'] = Variable(torch.from_numpy(self.data_pos2[current_index]).long())
        batch['mask'] = Variable(torch.from_numpy(self.data_mask[current_index]).long())
        batch['id'] = Variable(torch.from_numpy(self.uid[current_index]).long())
        batch['label']= Variable(torch.from_numpy(self.data_label[current_index]).long())


        for key in batch:
            batch[key] = batch[key].cuda()

        return batch
    def get_same_entpair_ins(self, entpair):
        '''
        return instances with the same entpair
        entpair: a string with the format '$head_entity#$tail_entity'
        '''
        if not entpair in self.entpair2scope:
            return None
        scope = self.entpair2scope[entpair]
        batch = {}
        batch['word'] = Variable(torch.from_numpy(self.data_word[scope]).long())
        batch['pos1'] = Variable(torch.from_numpy(self.data_pos1[scope]).long())
        batch['pos2'] = Variable(torch.from_numpy(self.data_pos2[scope]).long())
        batch['mask'] = Variable(torch.from_numpy(self.data_mask[scope]).long())
        batch['label']= Variable(torch.from_numpy(self.data_label[scope]).long())
        batch['id'] = Variable(torch.from_numpy(self.uid[scope]).long())
        batch['entpair'] = [entpair] * len(scope)

        # To cuda
        for key in ['word', 'pos1', 'pos2', 'mask', 'id']:
            batch[key] = batch[key].cuda()

        return batch

    def get_random_candidate(self, pos_class, num_class, num_ins_per_class):
        '''
        random pick some instances for snowball phase 2 with total number num_class (1 pos + num_class-1 neg) * num_ins_per_class
        pos_class: positive relation (name)
        num_class: total number of classes, including the positive and negative relations
        num_ins_per_class: the number of instances of each relation
        return: a dataset
        '''

        target_classes = random.sample(self.rel2scope.keys(), num_class)

        if not pos_class in target_classes:
            target_classes = target_classes[:-1] + [pos_class]

        candidate = {'word': [], 'pos1': [], 'pos2': [],'mask': [], 'id': [], 'entpair': []}

        for i, class_name in enumerate(target_classes):
            if type(class_name) == list:
                class_name = class_name[0]

            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), min(num_ins_per_class, scope[1] - scope[0]),
                                       False)
            candidate['word'].append(self.data_word[indices])
            candidate['pos1'].append(self.data_pos1[indices])
            candidate['pos2'].append(self.data_pos2[indices])
            candidate['mask'].append(self.data_mask[indices])
            candidate['id'].append(self.uid[indices])
            candidate['entpair'] += list(self.data_entpair[indices])

        candidate['word'] = np.concatenate(candidate['word'], 0)
        candidate['pos1'] = np.concatenate(candidate['pos1'], 0)
        candidate['pos2'] = np.concatenate(candidate['pos2'], 0)
        candidate['mask'] = np.concatenate(candidate['mask'], 0)
        candidate['id'] = np.concatenate(candidate['id'], 0)

        candidate['word'] = Variable(torch.from_numpy(candidate['word']).long())
        candidate['pos1'] = Variable(torch.from_numpy(candidate['pos1']).long())
        candidate['pos2'] = Variable(torch.from_numpy(candidate['pos2']).long())
        candidate['mask'] = Variable(torch.from_numpy(candidate['mask']).long())
        candidate['id'] = Variable(torch.from_numpy(candidate['id']).long())


        for key in ['word', 'pos1', 'pos2', 'mask', 'id']:
            candidate[key] = candidate[key].cuda()

        return candidate