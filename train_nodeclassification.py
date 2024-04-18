import time
import torch
from utils import Logger
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import f1_score
from sklearn import svm
def extract_feature_list_layer2(feature_list):
    xx_list = []
    xx_list.append(feature_list[-1])
    tmp_feat = torch.cat(feature_list, dim=-1)
    xx_list.append(tmp_feat)
    return xx_list
def test_classify(feature, labels):
    f1_mac = []
    f1_mic = []
    accs = []
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(feature):
        train_X, train_y = feature[train_index], labels[train_index]
        test_X, test_y = feature[test_index], labels[test_index]
        clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
        clf.fit(train_X, train_y)
        preds = clf.predict(test_X)

        micro = f1_score(test_y, preds, average='micro')
        macro = f1_score(test_y, preds, average='macro')
        correct = (preds == test_y).astype(float)
        correct = correct.sum()
        acc = correct / len(test_y)
        accs.append(acc)
        f1_mac.append(macro)
        f1_mic.append(micro)
    f1_mic = np.array(f1_mic)
    f1_mac = np.array(f1_mac)
    accs = np.array(accs)
    f1_mic = np.mean(f1_mic)
    f1_mac = np.mean(f1_mac)
    accs = np.mean(accs)
    return f1_mic, f1_mac, accs
def train(model, data_a,splits, args, device="cpu"):

    def train(data):
        model.train()
        loss = model.train_epoch(data.to(device), optimizer,
                                 alpha=args.alpha, batch_size=args.batch_size)
        return loss

    @torch.no_grad()
    def test(splits, batch_size=2**16):
        model.eval()
        train_data = splits['train'].to(device)
        z = model(train_data.x, train_data.edge_index)

        valid_auc, valid_ap = model.test(
            z, splits['valid'].pos_edge_label_index, splits['valid'].neg_edge_label_index, batch_size=args.batch_size)

        test_auc, test_ap = model.test(
            z, splits['test'].pos_edge_label_index, splits['test'].neg_edge_label_index, batch_size=args.batch_size)

        results = {'AUC': (valid_auc, test_auc), 'AP': (valid_ap, test_ap)}
        return results

    monitor = 'AUC'
    save_path = args.save_path
    runs = 3
    loggers = {
        'AUC': Logger(runs, args),
        'AP': Logger(runs, args),
    }
    print('Start Training (Link Prediction Pretext Training)...')

    out2_dict = {0: 'last', 1: 'combine'}
    result_dict = out2_dict
    svm_result_final = np.zeros(shape=[runs, len(out2_dict)])
    for run in range(runs):
        model.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

        best_valid = 0.0
        best_epoch = 0
        cnt_wait = 0
        for epoch in range(1, 1 + args.epochs):

            t1 = time.time()
            loss = train(splits['train'])
            t2 = time.time()

            if epoch % args.eval_period == 0:
                results = test(splits)

                valid_result = results[monitor][0]
                if valid_result > best_valid:
                    best_valid = valid_result
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path)
                    cnt_wait = 0
                else:
                    cnt_wait += 1

                for key, result in results.items():
                    valid_result, test_result = result
                    print(key)
                    print(f'Run: {run + 1:02d} / {args.runs:02d}, '
                          f'Epoch: {epoch:02d} / {args.epochs:02d}, '
                          f'Best_epoch: {best_epoch:02d}, '
                          f'Best_valid: {best_valid:.2%}%, '
                          f'Loss: {loss:.4f}, '
                          f'Valid: {valid_result:.2%}, '
                          f'Test: {test_result:.2%}',
                          f'Training Time/epoch: {t2-t1:.3f}')
                print('#' * round(140*epoch/(args.epochs+1)))
                if cnt_wait == args.patience:
                    print('Early stopping!')
                    break
        print('##################################### Testing on {}/{} #############################################'.format(run + 1, runs))

        model.load_state_dict(torch.load(save_path))
        results = test(splits, model)

        for key, result in results.items():
            valid_result, test_result = result
            print(f'{key} Testing on Run: '
                  f'Best Epoch: {best_epoch:02d}, '
                  f'Valid: {valid_result:.2%}, '
                  f'Test: {test_result:.2%}')
            if args.writer:
                fh = open('./log/{}_{}.txt'.format(args.dataset, args.layer), 'a')
                fh.write(f'{key} Testing on Run: '
                      f'Best Epoch: {best_epoch:02d},'
                      f'Valid: {valid_result:.2%}, '
                      f'Test: {test_result:.2%}')
                fh.write('\r\n')
                fh.flush()
                fh.close()

        for key, result in results.items():
            loggers[key].add_result(run, result)

#####################################svm分类############################################
        data_a = data_a.to(device)
        y = data_a.y.squeeze()
        embedding, out = model.encoder.get_embedding(data_a.x, data_a.edge_index, l2_normalize=args.l2_normalize)
        feature = [feature.detach() for feature in out]
        feature_list = extract_feature_list_layer2(feature)
        # X_train, X_test, Y_train, Y_test = train_test_split(embedding.cpu().detach().numpy(),
        #                                                     y.cpu().detach().numpy(), test_size=0.2, random_state=0)
        # svc = svm.SVC(probability=True)
        # svc.fit(X_train, Y_train)
        # Pred_Y = svc.predict(X_test)
        # correct = (Pred_Y == Y_test).astype(float)
        # correct = correct.sum()
        # acc = correct / len(Y_test)
        # print('acc:', acc)
# ########################################################################################
        for i, feature in enumerate(feature_list):
            feature = feature.cpu().detach().numpy()
            labels = y.cpu().detach().numpy()
            f1_mic_svm, f1_mac_svm, acc_svm = test_classify(feature, labels)

            svm_result_final[run, i] = acc_svm
            print('SVM test acc on Run for {} is F1-mic={} F1-mac={} acc={}'
                  .format(result_dict[i], f1_mic_svm, f1_mac_svm, acc_svm))
    svm_result_final = np.array(svm_result_final)
    print('\n--------------------- Print final result for SVM---------------------------')
    for i in range(len(out2_dict)):
        temp_resullt = svm_result_final[:, i]
        print('Final svm test result on {} is mean={} std={}'.format(result_dict[i], np.mean(temp_resullt),
                                                                          np.std(temp_resullt)))
########################################################################################

    print('\n---------- Final Testing result (Link Prediction Pretext Training)-------------')
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()

    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print('Successfully delete the saved models')



