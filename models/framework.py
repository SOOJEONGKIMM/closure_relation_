import os
import torch
from torch import optim
import torch.nn as nn
import sys
from torch.autograd import Variable
import random
import copy

from transformers import BertTokenizer, BertModel, BertForMaskedLM
class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path).cuda()

    def forward(self, inputs):
        _, x = self.bert(inputs['word'], attention_mask=inputs['mask'], return_dict=False)
        return x
class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, distant, args=None):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.args = args
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        #from snowball
        self.neg_loader = train_data_loader
        self.distant = distant
        self.sentence_encoder = BERTSentenceEncoder('D:/2022_2_GraphTransformer\graphtransformer/Neural-Snowball_\data/bert-base-uncased')
        #self.sentence_encoder = self.sentence_encoder.cuda()
        self.hidden_size = 200
        self.fc = nn.Linear(self.hidden_size, 1)
        self.euc = True
        self.sort_num1=5
        self.sort_threshold1=0.5
        self.sort_num2 = 5
        self.sort_threshold2 = 0.5
        self.sort_ori_threshold = 0.9
        drop_rate = 0.5
        self.drop = nn.Dropout(drop_rate)
        self.pos_class = None
        self.model = None
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    def encode(self, dataset, batch_size=4):
        self.sentence_encoder.eval()
        with torch.no_grad():
            #if self.pre_rep is not None:
             #   return self.pre_rep[dataset['id'].view(-1)]

            if batch_size == 0:
                x = self.sentence_encoder(dataset)
            else:
                total_length = dataset['word'].size(0)
                max_iter = total_length // batch_size
                if total_length % batch_size != 0:
                    max_iter += 1
                x = []
                for it in range(max_iter):
                    scope = list(range(batch_size * it, min(batch_size * (it + 1), total_length)))
                    with torch.no_grad():
                        _ = {'word': dataset['word'][scope], 'mask': dataset['mask'][scope]}
                        if 'pos1' in dataset:
                            _['pos1'] = dataset['pos1'][scope]
                            _['pos2'] = dataset['pos2'][scope]
                        _x = self.sentence_encoder(_)
                    x.append(_x.detach())
                x = torch.cat(x, 0)
            return x

    def _train_finetune_init(self):
        # init variables and optimizer
        self.new_W = Variable(self.fc.weight.mean(0) / 1e3, requires_grad=True)
        self.new_bias = Variable(torch.zeros((1)), requires_grad=True)
        self.optimizer = optim.Adam([self.new_W, self.new_bias], self.args.finetune_lr,
                                    weight_decay=self.args.finetune_wd)
        self.new_W = self.new_W.cuda()
        self.new_bias = self.new_bias.cuda()

    def _train_finetune(self, data_repre, learning_rate=None, weight_decay=1e-5):
        '''
        train finetune classifier with given data
        data_repre: sentence representation (encoder's output)
        label: label
        '''

        #self.train()

        optimizer = self.optimizer
        if learning_rate is not None:
            optimizer = optim.Adam([self.new_W, self.new_bias], learning_rate, weight_decay=weight_decay)

        # hyperparameters
        max_epoch = self.args.finetune_epoch
        batch_size = self.args.finetune_batch_size

        # dropout
        data_repre = self.drop(data_repre)

        # train
        for epoch in range(max_epoch):
            max_iter = data_repre.size(0) // batch_size
            if data_repre.size(0) % batch_size != 0:
                max_iter += 1
            order = list(range(data_repre.size(0)))
            random.shuffle(order)
            for i in range(max_iter):
                x = data_repre[order[i * batch_size: min((i + 1) * batch_size, data_repre.size(0))]]
                # batch_label = label[order[i * batch_size : min((i + 1) * batch_size, data_repre.size(0))]]

                # neg sampling
                # ---------------------
                batch_label = torch.ones((x.size(0))).long().cuda()
                neg_size = int(x.size(0) * 1)
                neg = self.neg_loader.neg_batch(neg_size)
                neg = self.encode(neg, self.args.infer_batch_size)
                x = torch.cat([x, neg], 0)
                batch_label = torch.cat([batch_label, torch.zeros((neg_size)).long().cuda()], 0)
                # ---------------------

                x = torch.matmul(x, self.new_W) + self.new_bias  # (batch_size, 1)
                x = torch.sigmoid(x)

                # iter_loss = self.__loss__(x, batch_label.float()).mean()
                weight = torch.ones(batch_label.size(0)).float().cuda()
                weight[batch_label == 0] = self.args.finetune_weight  # 1 / float(max_epoch)
                iter_loss = (self.__loss__(x, batch_label.float()) * weight).mean()

                optimizer.zero_grad()
                iter_loss.backward(retain_graph=True)
                optimizer.step()
                sys.stdout.write('[snowball finetune] epoch {0:4} iter {1:4} | loss: {2:2.6f}'.format(epoch, i,
                                                                                                          iter_loss) + '\r')
                sys.stdout.flush()
        self.eval()
    def _infer(self, dataset, batch_size=4):
        '''
        get prob output of the finetune network with the input dataset
        dataset: input dataset
        return: prob output of the finetune network
        '''
        x = self.encode(dataset, batch_size=batch_size)
        x = torch.matmul(x, self.new_W) + self.new_bias # (batch_size, 1)
        x = torch.sigmoid(x)
        return x.view(-1)
    def forward_infer_sort(self, x, y, batch_size=0):

        #x = self.encode(x, batch_size=batch_size)
        #y = self.encode(y, batch_size=batch_size)
        x, _, _ = self.model.context_encoder(x)
        y, _, _ = self.model.context_encoder(y)
        x = x[:, :min(x.size(1), y.size(1)), :]
        y = y[:, :min(x.size(1), y.size(1)), :]

        x = x.unsqueeze(1)
        y = y.unsqueeze(0)

        if self.euc:
            dis = torch.pow(x - y, 2)
            score = torch.sigmoid(self.fc(dis.cpu()).squeeze(-1)).mean(0)
        else:
            z = x * y
            z = self.fc(z.cpu()).squeeze(-1)
            score = torch.sigmoid(z).mean(0)

        pred = []
        for i in range(score.size(0)):
            pred.append((score[i].cpu(), i))

        pred.sort(key=lambda x: x[0][0], reverse=True)
        return pred[0][0]

    def _add_ins_to_data(self, dataset_dst, dataset_src, ins_id, label=None):
        '''
        add one instance from dataset_src to dataset_dst (list)
        dataset_dst: destination dataset
        dataset_src: source dataset
        ins_id: id of the instance
        '''

        dataset_dst['word'].append(dataset_src['word'][ins_id])

        if 'pos1' in dataset_src:
            if ins_id < len(dataset_src['pos1']):
                dataset_dst['pos1'].append(dataset_src['pos1'][ins_id])
                dataset_dst['pos2'].append(dataset_src['pos2'][ins_id])
            else:
                dataset_dst['pos1'].append(dataset_src['pos1'][len(dataset_src['pos1'])-1]) #error 임시방편
                dataset_dst['pos2'].append(dataset_src['pos2'][len(dataset_src['pos1'])-1])
        dataset_dst['mask'].append(dataset_src['mask'][ins_id])
        if 'id' in dataset_dst and 'id' in dataset_src:
            dataset_dst['id'].append(dataset_src['id'][ins_id])
        if 'entpair' in dataset_dst and 'entpair' in dataset_src:
            dataset_dst['entpair'].append(dataset_src['entpair'][ins_id])
        if 'label' in dataset_dst and label is not None:
            dataset_dst['label'].append(label)

    def _add_ins_to_vdata(self, dataset_dst, dataset_src, ins_id, label=None):
        '''
        add one instance from dataset_src to dataset_dst (variable)
        dataset_dst: destination dataset
        dataset_src: source dataset
        ins_id: id of the instance
        '''

        dataset_dst['word'] = torch.cat([dataset_dst['word'], dataset_src['word']], 0)
        if 'pos1' in dataset_src:
            dataset_dst['pos1'] = torch.cat([dataset_dst['pos1'], dataset_src['pos1']], 0)
            dataset_dst['pos2'] = torch.cat([dataset_dst['pos2'], dataset_src['pos2']], 0)
        dataset_dst['mask'] = torch.cat([dataset_dst['mask'], dataset_src['mask']], 0)
        if 'id' in dataset_dst and 'id' in dataset_src:
            dataset_dst['id'] = torch.cat([dataset_dst['id'], dataset_src['id']], 0)

        if 'entpair' in dataset_dst and 'entpair' in dataset_src:
            dataset_dst['entpair'].append(dataset_src['entpair'][0])
        if 'label' in dataset_dst and label is not None:
            dataset_dst['label'] = torch.cat([dataset_dst['label'], torch.ones((1)).long().cuda()], 0)

    def _dataset_stack_and_cuda(self, dataset):
        '''
        stack the dataset to torch.Tensor and use cuda mode
        dataset: target dataset
        '''
        if (len(dataset['word']) == 0):
            return
        dataset['word'] = torch.stack(dataset['word'], 0).cuda()
        if 'pos1' in dataset:
            dataset['pos1'] = torch.stack(dataset['pos1'], 0).cuda()
            dataset['pos2'] = torch.stack(dataset['pos2'], 0).cuda()
        dataset['mask'] = torch.stack(dataset['mask'], 0).cuda()
        dataset['id'] = torch.stack(dataset['id'], 0).cuda()
    def _forward_train(self, support_pos, label):
        '''
        snowball process (train)
        support_pos: support set (positive, raw data)
        support_neg: support set (negative, raw data)
        query: query set
        distant: distant data loader
        threshold: ins with prob > threshold will be classified as positive
        threshold_for_phase1: distant ins with prob > th_for_phase1 will be added to extended support set at phase1
        threshold_for_phase2: distant ins with prob > th_for_phase2 will be added to extended support set at phase2
        '''

        # hyperparameters
        snowball_max_iter = 5 #self.args.snowball_max_iter
        # snowball
        exist_id = {}
        # init
        self._train_finetune_init()
        support_pos_backup = copy.deepcopy(support_pos)
        #print(support_pos)

        #support_pos_rep = self.encode(support_pos, self.args.infer_batch_size)
        #print(support_pos_rep)
       # self._train_finetune(support_pos_rep)
       # print('\n-------------------------------------------------------')
        for snowball_iter in range(snowball_max_iter):
        #    print('###### snowball iter ' + str(snowball_iter))
            entpair_support = {}
            entpair_distant = {}
            for i in range(len(support_pos['entpair'])):  # only positive support
                ##if 'pos1' in support_pos:
                   ## if i==len(support_pos['pos1']):
                     ##   break
                entpair = support_pos['entpair'][i]

                exist_id[support_pos['id'][i]] = 1
                if entpair not in entpair_support:
                    if 'pos1' in support_pos:
                        entpair_support[entpair] = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': []}
                    else:
                        entpair_support[entpair] = {'word': [], 'mask': [], 'id': []}
                self._add_ins_to_data(entpair_support[entpair], support_pos, i)
            for entpair in entpair_support:
                raw = self.distant.get_same_entpair_ins(entpair)  # ins with the same entpair
                if raw is None:
                    continue
                if 'pos1' in support_pos:
                    entpair_distant[entpair] = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'id': [],
                                                'entpair': []}
                else:
                    entpair_distant[entpair] = {'word': [], 'mask': [], 'id': [], 'entpair': []}
                for i in range(raw['word'].size(0)):
                    if raw['id'][i] not in exist_id:  # don't pick sentences already in the support set
                        self._add_ins_to_data(entpair_distant[entpair], raw, i)
                self._dataset_stack_and_cuda(entpair_support[entpair])
                self._dataset_stack_and_cuda(entpair_distant[entpair])
                if len(entpair_support[entpair]['word']) == 0 or len(entpair_distant[entpair]['word']) == 0:
                    continue


                pick_or_not = self.forward_infer_sort(entpair_support[entpair], entpair_distant[entpair],
                                                                    batch_size=0)

                # pick_or_not = self.siamese_model.forward_infer_sort(original_support_pos, entpair_distant[entpair], threshold=threshold_for_phase1)
                # pick_or_not = self._infer(entpair_distant[entpair]) > threshold


                # -- method B: use sort --
                for i in range(min(len(pick_or_not), self.sort_num1)):
                    if pick_or_not[i] > self.sort_threshold1:
                        #iid = pick_or_not[i][1]

                        self._add_ins_to_vdata(support_pos, entpair_distant[entpair], i, label=1)

                        #print("exist_id:", exist_id)
                        #print("entpair_distant[entpair]['id']: ", entpair_distant[entpair]['id'])
                        if i>= len(entpair_distant[entpair]['id']):
                            break
                        exist_id[entpair_distant[entpair]['id'][i]] = 1

            #print(support_pos)
            '''
            support_pos_rep = self.encode(support_pos, batch_size=self.args.infer_batch_size)

            self._train_finetune_init()
         #   self._train_finetune(support_pos_rep)
            '''
            candidate_num_class = 10
            candidate_num_ins_per_class = 10
            candidate = self.distant.get_random_candidate(self.pos_class[0], candidate_num_class, candidate_num_ins_per_class)


            ### -- method 1: directly use the classifier --
            ##candidate_prob = self._infer(candidate, batch_size=self.args.infer_batch_size)
            ## -- method 2: use siamese network --

            pick_or_not = self.forward_infer_sort(support_pos_backup, candidate,
                                                                batch_size=self.args.infer_batch_size)


            self._phase2_total = candidate['word'].size(0)
            for i in range(self.sort_num2):
                #iid = pick_or_not[i][1]
                if (pick_or_not[i]> self.sort_threshold2) and not (
                        candidate['id'] in exist_id):
                    exist_id[candidate['id'][i]] = 1
                    self._add_ins_to_vdata(support_pos, candidate, i, label=1)
            '''
            ## build new support set
            support_pos_rep = self.encode(support_pos, self.args.infer_batch_size)

            ## finetune
            # print("Fine-tune Init")
            self._train_finetune_init()
          #  self._train_finetune(support_pos_rep)
        '''

        return support_pos

    def train(self, model, model_name, B=4, N_for_train=20, N_for_eval=5, K=5, Q=100,
              ckpt_dir='./checkpoint', learning_rate=1e-1, lr_step_size=20000,
              weight_decay=1e-5, train_iter=30000, val_iter=1000, val_step=2000,
              test_iter=3000, pretrain_model=None, optimizer=optim.SGD):
        '''
        model: model
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        test_result_dir: Directory of test results
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path
        '''
        # Init
        self.model = model
        parameters_to_optimize = filter(lambda x:x.requires_grad, model.parameters())
        optimizer = optimizer(parameters_to_optimize, learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)
        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_iter = checkpoint['iter'] + 1
        else:
            start_iter = 0

        model = model.cuda()
        model.train()

        # Training
        best_acc = 0
        import csv
        #f = open("5way-5shot_s.csv","a",encoding='cp949')
        #wr = csv.writer(f)
        #wr.writerow([supp['id'],supp_snow['id'],supp['entpair'],supp_snow['entpair']])

        for it in range(start_iter, start_iter + train_iter):

            scheduler.step()

            support, query, label, target_classes = self.train_data_loader.next_batch(B, N_for_train, K, Q)
            #support_ = copy.deepcopy(support)
            #print("support:", support['entpair'])
            #print("label:", label)
            self.pos_class = target_classes



            #MLMAN baseline model
            logits, pred, dist = model(support, query, N_for_train, K, Q)#, eval=False)

            # my model
            support_snowball = self._forward_train(support, label)
            ##support=0
            # print("next support shape:", support_['mask'].shape)
            #print("support_snowball:", support_snowball['entpair'])
            #f = open("5way-5shot_s.csv", "w", encoding='utf-8')
            #wr = csv.writer(f)
            #wr.writerow([support['entpair'],support_snowball['entpair']])
            # print("======it start_iter, start_iter + train_iter:", it, start_iter, start_iter + train_iter)
            logits_s, pred_s, dist_s = model(support_snowball, query, N_for_train, K, Q)#, eval=False)
            logits_1 = logits + logits_s
            loss = model.loss(logits_1, label)
            allloss = loss + dist + dist_s


            #loss = model.loss(logits, label)
            #allloss = loss + dist
            optimizer.zero_grad()
            allloss.backward()
            optimizer.step()

            if (it + 1) % val_step == 0:

                with torch.no_grad():

                    acc = self.eval(model, 5, N_for_eval, K, 5, val_iter)
                    print("{0:}---{1:}-way-{2:}-shot test   Test accuracy: {3:3.2f}".format(it, N_for_eval, K, acc*100))

                    if acc > best_acc:
                        print('Best checkpoint')
                        if not os.path.exists(ckpt_dir):
                            os.makedirs(ckpt_dir)
                        save_path = os.path.join(ckpt_dir, model_name + ".pth.tar")
                        torch.save({'state_dict': model.state_dict()}, save_path)
                        best_acc = acc

                model.train()


        print("\n####################\n")
        print("Finish training " + model_name)
        with torch.no_grad():
            test_acc = self.eval(model, 5, N_for_eval, K, 5, test_iter, ckpt=os.path.join(ckpt_dir, model_name + '.pth.tar'))
            print("{0:}-way-{1:}-shot test   Test accuracy: {2:3.2f}".format(N_for_eval, K, test_acc*100))


    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            ckpt=None):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            checkpoint = self.__load_model__(ckpt)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            eval_dataset = self.test_data_loader
        model.eval()

        iter_right = 0.0
        iter_sample = 0.0
        for it in range(eval_iter):
            support, query, label, target_classes = eval_dataset.next_batch(B, N, K, Q)
            #support_ = copy.deepcopy(support)
            # print("support_snowball:", support_snowball)
            ##_, pred, _  = model(support_snowball, query, N, K, Q, eval=True)
            ##_, pred, _ = model(support, query, N, K, Q)#, eval=True)
            #print("N K Q:", N, K, Q)
            logits_, pred_, _ = model(support, query, N, K, Q, eval=True)
            support_snowball = self._forward_train(support, label)
            logits_s, pred_s, dist_s = model(support_snowball, query, N, K, Q, eval=True)
            _, pred = torch.max(logits_+logits_s, dim=1)
            right = model.accuracy(pred, label)
            iter_right += right.item()
            iter_sample += 1
        return iter_right / iter_sample
