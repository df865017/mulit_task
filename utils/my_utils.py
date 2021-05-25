#!/usr/bin/env python
# coding=utf-8

import datetime
import os


def arg_parse(argv):
    parse_dict = dict()
    # for i in range(1, len(argv)):
    #     line_parse = argv[i].split("=")
    #     key = line_parse[0].strip()
    #     value = line_parse[1].strip()
    #     parse_dict[key] = value
    return parse_dict


def shift_date_time(dt_time, offset_day, time_structure='%Y%m%d'):
    dt = datetime.datetime(int(dt_time[0:4]), int(dt_time[4:6]),
                           int(dt_time[6:8]))
    delta = datetime.timedelta(days=offset_day)
    del_day_date = dt + delta
    del_day_time = del_day_date.strftime(time_structure)
    return del_day_time


def shift_hour_time(dt_time, offset_hour, time_structure='%Y%m%d%H'):
    dt = datetime.datetime(int(dt_time[0:4]), int(dt_time[4:6]),
                           int(dt_time[6:8]), int(dt_time[8:10]))
    delta = datetime.timedelta(hours=offset_hour)
    del_date = dt + delta
    del_time = del_date.strftime(time_structure)
    return del_time


def feat_size(path, alg_name):
    cont_size = 0
    vector_size = 0
    cate_size = 0
    multi_feats_size = 0
    multi_cate_field = 0
    attention_feats_size = 0
    multi_cate_range = []
    cate_range = []
    attention_range = []
    attention_cate_field = []
    vec_name_list = ["user_vec", "ruUserVec", "item_vec", "user_kgv", "item_kgv"]
    no_pool_alg = ["dnn", "deepfm"]
    attention_alg = ["din", "dinfm", "dien"]

    files = os.listdir(path)
    for file in files:
        if file != "dnn.conf" and file != "lr.conf":
            continue

        file_path = path + "/" + file
        print("----read %s----" % file_path)
        with open(file_path, 'r') as f:
            index_start = 0
            for line in f.readlines():
                line_data = line.strip()
                if line_data == '':
                    continue

                try:
                    config_arr = line_data.split("\t")
                    col_name = config_arr[0]
                    result_type = config_arr[2]
                    result_parse_type = config_arr[6]
                    result_parse = config_arr[7]
                    is_drop = int(config_arr[8])
                    feature_name = config_arr[9]
                    is_attention = int(config_arr[10])

                    if is_drop == 1:
                        continue

                    if result_type == 'vector' or result_type == 'vec':
                        if col_name in vec_name_list:
                            vector_size += 200
                        else:
                            print("%s is error" % line)
                            exit(-1)

                    elif result_type == 'arr':
                        if result_parse_type == 'top' or result_parse_type == 'top_arr':
                            top_n = int(result_parse.strip().split("=")[1])
                            size = 1
                        elif result_parse_type == 'top_multi':
                            parse_arr = result_parse.split(";")
                            top_n = int(parse_arr[0].split("=")[1])
                            size = int(parse_arr[1].split("=")[1])
                        else:
                            print("%s is error" % line)
                            exit(-1)

                        if alg_name not in no_pool_alg:
                            index_end = index_start + top_n * size
                            index_range = [index_start, index_end, feature_name, col_name]
                            index_start = index_end
                            if is_attention == 1 and alg_name in attention_alg:
                                attention_feats_size += top_n * size
                                attention_cate_field.append(index_range)
                            else:
                                multi_cate_field += 1
                                multi_feats_size += top_n * size
                                multi_cate_range.append(index_range)
                        else:
                            cate_size += top_n * size

                    elif result_type == 'string':
                        cate_index_name = [cate_size, feature_name, col_name]
                        cate_range.append(cate_index_name)
                        cate_size += 1
                    elif result_type == 'float':
                        cont_size += 1
                    else:
                        print("%s is error!!!" % line_data)
                except Exception as e:
                    print("-----------feat_conf is Error!!!!-----------")
                    print(e)
                    print(line_data)
                    exit(-1)

    # get attention range
    for attention_cate in attention_cate_field:
        cate_match = False
        for cate in cate_range:
            if attention_cate[-2] == cate[-2] and cate[-1][:4] == 'item':
                match_tuple = (0, attention_cate[-2], cate[0], (attention_cate[0], attention_cate[1]))
                attention_range.append(match_tuple)
                cate_match = True
                break
        if not cate_match:
            for m_c in multi_cate_range:
                if attention_cate[-2] == m_c[-2] and m_c[-1][:4] == 'item':
                    match_tuple = (1, m_c[-2], (m_c[0], m_c[1]), (attention_cate[0], attention_cate[1]))
                    attention_range.append(match_tuple)
                    break

    return cont_size, vector_size, cate_size, multi_feats_size, multi_cate_range, attention_feats_size, attention_range
