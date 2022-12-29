import os
import re
from tqdm import tqdm

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

with open("result.txt", "w", encoding="utf-8") as f:
    f.writelines(rm_noug(os.path.join(os.getcwd(),"data","raw")))
   
