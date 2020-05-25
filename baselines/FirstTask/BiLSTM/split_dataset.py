import pandas as pd

def split():
    s1_dict = {}
    data = pd.read_csv('./data/sentpair/cmp0516_5_bk.csv')
    # id = data[" "]
    all_s1 = data["sent01"]
    all_s2 = data["sent02"]
    all_label01 = data["label01"]
    all_label02 = data["label02"]
    all_label = data["cmp"]
    for s1,s2,l1,l2,l in zip(all_s1,all_s2,all_label01,all_label02,all_label):
        if s1 in s1_dict:
            s1_dict[s1] += [(s2,l1,l2,l)]
        else:
            s1_dict[s1] = [(s2,l1,l2,l)]
    print(len(s1_dict))
    i = 0
    s1_list,s2_list,l1_list,l2_list,label_list = [],[],[],[],[]
    dev_s1_list,dev_s2_list,dev_l1_list,dev_l2_list,dev_label_list = [],[],[],[],[]
    test_s1_list,test_s2_list,test_l1_list,test_l2_list,test_label_list = [],[],[],[],[]
    for k,v_list in s1_dict.items():
        i += len(v_list)
        if i < 27868:
            for tup in v_list:
                s1_list.append(k)
                s2_list.append(tup[0])
                label_list.append(tup[-1])
                l1_list.append(tup[1])
                l2_list.append(tup[2])
        if i>= 27868 and i < 31351:
            for tup in v_list:
                dev_s1_list.append(k)
                dev_s2_list.append(tup[0])
                dev_label_list.append(tup[-1])
                dev_l1_list.append(tup[1])
                dev_l2_list.append(tup[2])
        if i >= 31351 and i < 34834:
            for tup in v_list:
                test_s1_list.append(k)
                test_s2_list.append(tup[0])
                test_label_list.append(tup[-1])
                test_l1_list.append(tup[1])
                test_l2_list.append(tup[2])               
    print("train len:",len(s1_list))
    print("dev len:",len(dev_s1_list))
    print("test len",len(test_s1_list))
    train_dataframe = pd.DataFrame({'sent01': s1_list,'sent02': s2_list,'label01': l1_list,'label02': l2_list,"cmp":label_list})
    train_dataframe.to_csv("./data/sentpair/new_train.csv", sep=',', columns=['sent01','sent02','label01','label02',"cmp"])
    dev_dataframe = pd.DataFrame(
        {'sent01': dev_s1_list, 'sent02': dev_s2_list, 'label01': dev_l1_list, 'label02': dev_l2_list, "cmp": dev_label_list})
    dev_dataframe.to_csv("./data/sentpair/new_dev.csv", sep=',', columns=['sent01','sent02','label01','label02',"cmp"])
    test_dataframe = pd.DataFrame(
        {'sent01': test_s1_list, 'sent02': test_s2_list, 'label01': test_l1_list, 'label02': test_l2_list,
         "cmp": test_label_list})
    test_dataframe.to_csv("./data/sentpair/new_test.csv", sep=',', columns=['sent01','sent02','label01','label02',"cmp"])

if __name__ == "__main__":
    split()


