import json
import re
from tqdm import tqdm
from transformers import BloomTokenizerFast
import os
import pynlpir
from gensim import models
import gensim.corpora as corpora

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def convert_tokens_to_string(byte_decoder, tokens):
    """Converts a sequence of tokens (string) in a single string."""
    text = "".join(tokens)
    text_utf = bytearray([byte_decoder[c] for c in text]).decode("utf-8", 'ignore')
    return text_utf


# 根据路径中分词的文件，生成词典
def generate_tf_idf_dic(document_path:str) -> None:
    with open(document_path,"r",encoding="utf-8") as f:
        string_list100 = [i.strip().split(" ") for i in f.readlines()]
    id2word = corpora.Dictionary(string_list100)  # 创建词典
    id2word.filter_extremes(no_below=1, no_above=1, keep_n=100000)  # 筛选词典，出现次数至少为1，至少在一篇文档中出现，词典大小最多100000个词
    id2word.save(document_path + "_tfidf_dic")  # 保存词典
    pass

# 根据路径保存词与文档的一个共现矩阵
# tfidf_model = models.TfidfModel.load('test_tfidf.model') 载入模型
# corpus_tfidf = [tfidf_model[doc] for doc in corpus]
# print(corpus_tfidf[:1])
# print([[(id2word[id], freq) for id, freq in cp] for cp in corpus_tfidf[:1]])
def generate_tf_idf_model(document_path, dic_path:str) -> None:
    with open(document_path,"r",encoding="utf-8") as f:
        string_list100 = [i.strip().split(" ") for i in f.readlines()]
    id2word = corpora.Dictionary() 
    id2word.load(dic_path)
    corpus = [id2word.doc2bow(text) for text in string_list100]   # 分别对每篇文章建立词袋向量
    tfidf_model = models.TfidfModel(corpus=corpus, dictionary=id2word)
    tfidf_model.save(document_path + '_tfidf_model') #保存模型到本地


# 采用BloomTokenizerFast的bbpe分词藏语
def bbpe_seg_ti(ti_file: str):

    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    tokenizer = BloomTokenizerFast.from_pretrained("bloom-560m")
    utf_vocab = {}
    seg_sen = []
    for key,value in tokenizer.vocab.items():
        key_unicode = convert_tokens_to_string(byte_decoder, key)
        utf_vocab[str(value)] = key_unicode
    tokenizer = BloomTokenizerFast.from_pretrained("bloom-560m")
    with open(ti_file,"r",encoding="utf-8") as f:
        tis = f.readlines()
    ti_file_sen = ti_file+"sen"
    seg_word_f = {}
    with open(ti_file_sen,"a",encoding="utf-8") as f:
        f.truncate(0)
        for tis_i in tqdm(tis):
            seg_nlpir = ["  ".join(pynlpir.segment(i.strip(),pos_tagging=False)).split("  ") for i in zh]
            seg_id =  tokenizer(tis_i.strip())["input_ids"]
            seg_word = [utf_vocab[str(j)] for j in seg_id]
            for word in seg_word:
                if word in seg_word_f:
                    seg_word_f[word] += 1
                else:
                    seg_word_f[word] = 1
            seg_sen.append("  ".join([utf_vocab[str(j)] for j in seg_id])+"\n")
        f.writelines(seg_sen)
    
    # 保存词频词典并做图
    ti_file_sen_f_path = ti_file_sen + "f"
    my_dict = dict(sorted(seg_word_f.items(), key=lambda x: x[1], reverse=False))
    keys = my_dict.keys()
    values = my_dict.values()

    with open(ti_file_sen_f_path,"a",encoding="utf-8") as f:
        json.dump(my_dict,f,ensure_ascii=False)

# 判断数据是否存在空行
def detection_zh(path_ti:str, path_zh:str) -> None:
    with open(path_ti, "r", encoding="utf-8") as f_ti, open(path_zh, "r", encoding="utf-8") as f_zh:
        f_ti = f_ti.readlines()
        f_zh = f_zh.readlines()
    
    result_ti = []
    result_zh = []
    for i_ti, i_zh in tqdm(zip(f_ti, f_zh)):
        if i_ti == "\n" or i_zh == "\n":
            pass
        else:
            result_ti.append(i_ti)
            result_zh.append(i_zh)
    
    with open(path_ti + "_", "w", encoding="utf-8") as f_ti, open(path_zh + "_", "w", encoding="utf-8") as f_zh:
        f_ti.writelines(result_ti)
        f_zh.writelines(result_zh)
# "/data1/2023/yrh/datautils/data/raw/valid/ti.txt", "/data1/2023/yrh/datautils/data/raw/valid/zh.txt"
detection_zh("/data1/2023/yrh/datautils/data/raw/train/ti.txt", "/data1/2023/yrh/datautils/data/raw/train/zh.txt")


def nlpir_seg_zh(path_zh: str, path_out):
    '''
    将输入的中文使用nlpir分词, 并输出为一个分词的文件，每行为一个句子。
    input: 中文文件路径，其中每行为一个句子。
    output: None
    '''
    pynlpir.open()
    seg_nlpir = []
    with open(path_zh,"r",encoding="utf-8") as f:
        zh = f.readlines()
    for i in tqdm(zh):
        seg_nlpir.append(" ".join(pynlpir.segment(i.strip(),pos_tagging=False))+"\n")
    with open(path_out, "w",encoding="utf-8") as f:
        f.writelines(seg_nlpir)


# 统计分词词频
def nlpir_seg_small(data_file:str, generate_file:str):
    size = os.path.getsize(data_file)    
    size_point = 0
    seg_word_f = {}
    with open(generate_file,"a",encoding="utf-8") as f:
        f.truncate(0)

    with open(data_file,"r",encoding="utf-8") as f:
        f = f.readlines()
    for f_i, f_sen in tqdm(enumerate(f)):
        pynlpir.open()
        try:
            seg_nlpir = "  ".join(pynlpir.segment(f_sen.strip(),pos_tagging=False)).split("  ")
        except:
            continue
        for word in seg_nlpir:
            if word in seg_word_f:
                seg_word_f[word] += 1
            else:
                seg_word_f[word] = 1
            # seg_sen.append("  ".join(pynlpir.segment(tis_i.strip(),pos_tagging=False))+"\n")
    dict_ = dict(sorted(seg_word_f.items(), key=lambda x: x[1], reverse=True))
    with open(generate_file,"a",encoding="utf-8") as f:
        json.dump(dict_, f, ensure_ascii=False)    


# 统计nlpir中文前n个单词出现的次数占总单词出现次数的比例
def dic_fre(path_dic:str, path_out:str):
    with open(path_dic,"r",encoding="utf-8") as f:
        dic = json.load(f)
    dic_out = {}
    total = sum(dic.values())
    index = 1
    pre_num = 0
    for dic_zh,dic_fre in dic.items():
        pre_num += dic_fre
        dic_out[index] = pre_num / total
        index += 1
    with open(path_out,"w",encoding="utf-8") as f:
        json.dump(dic_out, f, ensure_ascii=False)    


# 统计bpe中文前n个单词出现的次数占总单词出现次数的比例 读取 每行 词:频次
def dic_fre_bpe(path_dic:str, path_out:str):
    with open(path_dic,"r",encoding="utf-8") as f:
        dic = [[i.strip().split(" ")[0], int(i.strip().split(" ")[1])] for i in f.readlines()]

    dic_out = {}
    total = sum([i[1] for i in dic])
    index = 1
    pre_num = 0
    for i in dic:
        pre_num += i[1]
        dic_out[index] = pre_num / total
        index += 1
    with open(path_out,"w",encoding="utf-8") as f:
        json.dump(dic_out, f, ensure_ascii=False)    


# 统计分词词频
def nlpir_seg_ti(data_file:str, generate_file:str):
    size = os.path.getsize(data_file)    
    size_point = 0
    seg_word_f = {}
    with open(generate_file,"a",encoding="utf-8") as f:
        f.truncate(0)
    while size_point < size:
        print("---"+str(size)+":"+str(size_point)+":"+str(size-size_point)+"---\n")
        # seg_sen = []
        with open(data_file,"r",encoding="utf-8") as f:
            f.seek(size_point)
            try:
                tis = f.read(size_point+1000000)
            except:
                size_point+=1
                continue
            size_point += 1000000
        for tis_i in tis.strip().split("\n"):
            pynlpir.open()
            try:
                seg_nlpir = "  ".join(pynlpir.segment(tis_i.strip(),pos_tagging=False)).split("  ")
            except:
                continue
            for word in seg_nlpir:
                if word in seg_word_f:
                    seg_word_f[word] += 1
                else:
                    seg_word_f[word] = 1
            # seg_sen.append("  ".join(pynlpir.segment(tis_i.strip(),pos_tagging=False))+"\n")
    dict_ = dict(sorted(seg_word_f.items(), key=lambda x: x[1], reverse=True))
    with open(generate_file,"a",encoding="utf-8") as f:
        json.dump(dict_, f, ensure_ascii=False) 

    keys = dict_.keys()
    values = dict_.values()    
    # 绘制柱状图
    plt.bar(keys, values)
    # 设置横轴标签和纵轴标签
    plt.xlabel('Fruit')
    plt.ylabel('Quantity')
    # 设置标题
    plt.title('Quantity of Fruits')
    # 展示图形
    plt.savefig('nlpir-zh-Hans.txt.png')

# 计算不同出现次数在词典中的占比。
def percent_fre(data_file, dic_filem, out_file) -> None:
    with open(data_file,"r",encoding="utf-8") as f_segtation:
        f_segtation = f_segtation.readlines()
    with open(dic_file,"r",encoding="utf-8") as f_dic:
        f_dic = json.load(f_dic)
    total_len = len(f_dic.items())
    result_dic = {}
    for f_dic_i, f_dic_num in f_dic.items():
        if f_dic_num in result_dic.keys():
            result_dic[f_dic_num] += 1
        else:
            result_dic[f_dic_num] = 1
    result_dic = dict(sorted(result_dic.items(), key=lambda x:x[0], reverse=True))
    for f_dic_i, f_dic_num in result_dic.items():
        result_dic[f_dic_i] = f_dic_num / total_len
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_dic, f)


# 计算词语所在的比例，多少词可以占满所有句子。
def percent(data_file, dic_file) -> None:
    with open(data_file,"r",encoding="utf-8") as f_segtation:
        f_segtation = f_segtation.readlines()
    with open(dic_file,"r",encoding="utf-8") as f_dic:
        f_dic = json.load(f_dic)
    result_dic = {}
    dic_list = []
    word_num = 1
    for f_dic_i, f_dic_j in f_dic.items():
        print(word_num)
        total_len = len(f_segtation)
        in_len = 0
        dic_list.append(f_dic_i)
        for sen_i in f_segtation:
            for dic_list_i in dic_list:
                if dic_list_i in sen_i:
                    in_len += 1
                    break
                else:
                    if in_len / total_len > 0.98:
                        print(sen_i)
        result_dic[str(word_num)] = in_len / total_len
        if in_len / total_len > 0.99:
            break
        word_num += 1
    with open("/data1/2023/yrh/k-translation/data/percent.txt", "w", encoding="utf-8") as f:
        json.dump(result_dic, f)


# 生成带提示模板的训练语料
def segCorpus_promptCorpus_sen(document_path_ti:str, document_path_zh:str, tfidf_dic_path:str, tfidf_model_path:str) -> None:
    
    with open(document_path_zh,"r",encoding="utf-8") as f:
        document_list_zh = f.readlines()
    with open(document_path_ti,"r",encoding="utf-8") as f:
        document_list = [i.strip().split(" ") for i in f.readlines()]

    id2word = corpora.Dictionary(document_list)  # 创建词典
    id2word.filter_extremes(no_below=1, no_above=1, keep_n=100000)  # 筛选词典，出现次数至少为1，至少在一篇文档中出现，词典大小最多100000个词
    # id2word = corpora.Dictionary() 
    # id2word.load(tfidf_dic_path)
    corpus = [id2word.doc2bow(text) for text in document_list]   # 分别对每篇文章建立词袋向量
    tfidf_model = models.TfidfModel(corpus=corpus, dictionary=id2word)
    # tfidf_model = models.TfidfModel.load(tfidf_model_path) # 载入模型
    corpus_tfidf = [tfidf_model[doc] for doc in corpus]
    prompt_ti = []
    for i, corpus_tfidf_i in tqdm(enumerate(corpus_tfidf)):
        flag = 0
        max_index = [l for k,l in corpus_tfidf_i].index(max([l for k,l in corpus_tfidf_i]))
        for j, document_list_j in enumerate(document_list):
            if document_list[i][max_index] in document_list_j and i!=j:
                flag += 1
                prompt_ti.append("  ".join(document_list_j) + " : " + document_list_zh[j].strip() + " | " + "  ".join(document_list[i]) + " : " + "\n")
                break
        if flag == 1:
            pass  
        else:
            prompt_ti.append("  ".join(document_list[i]) + " : " + document_list_zh[j] + " | " + "  ".join(document_list[i]) + " : " + "\n")
    with open(document_path_ti+"_prompt", "w", encoding="utf-8") as f:
        f.writelines(prompt_ti)
    pass


# 生成带单词提示模板的训练语料
def prompt_words(path_seg_ti:str, path_seg_zh:str, path_dic_bpe___:str, path_dic_ti_zh, path_out) -> None:
    with open(path_seg_ti,"r",encoding="utf-8") as f:
        path_seg_ti = f.readlines()
    with open(path_seg_zh,"r",encoding="utf-8") as f:
        path_seg_zh = [i.strip().split(" ") for i in f.readlines()]
    with open(path_dic_bpe___,"r",encoding="utf-8") as f:
        path_dic_bpe___ = json.load(f)
    with open(path_dic_ti_zh,"r",encoding="utf-8") as f:
        path_dic_ti_zh = json.load(f)
    
    # 颠倒key与value
    path_dic_zh_ti = dict([[value, key] for key, value in path_dic_ti_zh.items()])
    prompt = []

    for i, sen in enumerate(path_seg_zh):
        flag = 0
        prompt_sen = ""
        for word in sen:
            try:
                if path_dic_bpe___[word] > 8047:
                    flag = 1
                    if word in path_dic_ti_zh.values():
                        prompt_sen = prompt_sen + path_dic_zh_ti[word] + " " + word + " ; "
            except:
                pass
        prompt_sen = prompt_sen + path_seg_ti[i]
        prompt.append(prompt_sen)
    with open(path_out, "w", encoding="utf-8") as f:
        f.writelines(prompt)


# 将字典的value转换为它的排序序号
def serial(path_dic, path_out):
    with open(path_dic, "r", encoding="utf-8") as f:
        path_dic = json.load(f)
    flag = 1
    for key, value in path_dic.items():
        path_dic[key] = flag
        flag += 1
    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(path_dic, f, ensure_ascii=False)


# 根据分词语料生成词典
def seg_generate_dic(path_seg, path_out):
    with open(path_seg,"r",encoding="utf-8") as f:
        f = f.readlines()
    dic = {}
    for sen in f:
        sen = sen.strip().split(" ")
        for word in sen:
            if word in dic.keys():
                dic[word] += 1
            else:
                dic[word] = 1
    dic = dict(sorted(dic.items(), key = lambda x: x[1], reverse=True))
    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(dic, f, ensure_ascii=False)



# 清洗平行语料中可能存在的中文
def detection_zh(path_ti:str, path_zh:str) -> None:
    with open(path_ti, "r", encoding="utf-8") as f_ti, open(path_zh, "r", encoding="utf-8") as f_zh:
        f_ti = f_ti.readlines()
        f_zh = f_zh.readlines()
    
    result_ti = []
    result_zh = []
    for index, i_ti in tqdm(enumerate(f_ti)):
        flag = 0
        for j_i_ti in i_ti:
            if '\u4e00' <= j_i_ti <= '\u9fff':
                flag = 1
                break
        if flag == 1:
            continue
        else:
            result_ti.append(f_ti[index])
            result_zh.append(f_zh[index])
    
    with open(path_ti + "_", "w", encoding="utf-8") as f_ti, open(path_zh + "_", "w", encoding="utf-8") as f_zh:
        f_ti.writelines(result_ti)
        f_zh.writelines(result_zh)

detection_zh("/data1/2023/yrh/datautils/data/raw/valid/ti.txt", "/data1/2023/yrh/datautils/data/raw/valid/zh.txt")



# 替换多个匹配项
def rm_noug(dirpath):
    return_result = []
    for file in tqdm(os.listdir(dirpath)):
        with open(os.path.join(dirpath,file), "r", encoding="utf-8") as f:
            for line in f.readlines():
                ugPattern = re.compile(u'[\u0600-\u06FF]+')
                if ugPattern.search(line):
                    re1 = re.sub(r"[^\u0600-\u06FF0-9\–%]"," ",line)

                    re1 = re.sub("0"," 0 ",re1)
                    re1 = re.sub("1"," 1 ",re1)
                    re1 = re.sub("2"," 2 ",re1)
                    re1 = re.sub("3"," 3 ",re1)
                    re1 = re.sub("4"," 4 ",re1)
                    re1 = re.sub("5"," 5 ",re1)
                    re1 = re.sub("6"," 6 ",re1)
                    re1 = re.sub("7"," 7 ",re1)
                    re1 = re.sub("8"," 8 ",re1)
                    re1 = re.sub("9"," 9 ",re1)
                    re1 = re.sub("،"," ، ",re1)

                    re1 = re.sub(r" +"," ",re1)
                    return_result.append(re1.strip()+"\n")
    return return_result



if __name__ == '__main__':
    # data_file = "/data1/2023/yrh/k-translation/data/raw/train/zh.txt"
    # dic_file = "data/dic/zh_nlpir_dic.txt"
    # out_file = "/data1/2023/yrh/k-translation/data/dic/zh_nlpir_dic_precent_fre.txt"
    # generate_file = "data/dic/zh_nlpir_dic.txt"
    # bbpe_seg_ti(ti_file)

    # make_graph("data/raw/train/zh.txtsenf_bpe")
    # percent(data_file, dic_file)
    # percent_fre(data_file, dic_file, out_file)

    # data_file = "/data1/2023/yrh/k-translation/data/raw/train/zh.txt"
    # generate_file = "/data1/2023/yrh/k-translation/data/raw/train/zh_seg.txt"
    # nlpir_seg_zh(data_file, generate_file)

    # path_dic = "/data1/2023/yrh/tizh-test/data/dic_result/dic_bpe_@@.zh"
    # path_out = "/data1/2023/yrh/tizh-test/data/dic_result/dic_bpe_@@_fre.zh"
    # dic_fre_bpe(path_dic, path_out)

    # path_dic = "/data1/2023/yrh/tizh-test/data/dic_result/dic_bpe_@@.zh"
    # path_out = "/data1/2023/yrh/tizh-test/data/dic_result/dic_bpe_@@_fre.zh"
    # dic_fre(path_dic, path_out)

    # path_seg = "/data1/2023/yrh/tizh-test/data/final/train/train.ti.zh" 
    # path_out = "/data1/2023/yrh/tizh-test/data/dic_result/dic_bpe_@@.zh"
    # seg_generate_dic(path_seg, path_out)

    # path_dic = "/data1/2023/yrh/tizh-test/data/dic_result/dic_bpe_@@.zh"
    # path_out = "/data1/2023/yrh/tizh-test/data/dic_result/dic_bpe_@@_serial.zh"
    # serial(path_dic, path_out)

    # path_seg_ti = "/data1/2023/yrh/tizh-test/data/final/train/train.ti"
    path_seg_ti = "/data1/2023/yrh/tizh-test/data/raw/valid/ti.txt"
    path_seg_zh = "/data1/2023/yrh/tizh-test/data/raw/valid/zh.txt"
    path_dic_bpe___ = "/data1/2023/yrh/tizh-test/data/dic_result/dic_bpe_@@_serial.zh"
    path_dic_ti_zh = "/data1/2023/yrh/tizh-test/data/dic_zh_ti/dic_d221.txt"
    # path_out = "/data1/2023/yrh/tizh-test/data/final/train/train.promote"
    path_out = "/data1/2023/yrh/tizh-test/data/raw/valid/ti_promote.txt"
    prompt_words(path_seg_ti, path_seg_zh, path_dic_bpe___, path_dic_ti_zh, path_out)
