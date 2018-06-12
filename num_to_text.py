import nltk
import string
import re

def int_to_vn(num):
    d = {0: 'không', 1: 'một', 2: 'hai', 3: 'ba', 4: 'bốn', 5: 'năm', 6: 'sáu', 7: 'bảy', 8: 'tám', 9: 'chín', 10: 'mười'}
    if num <= 10: return d[num]
    if num//1000000 > 0:
        if num % 1000000 == 0: return int_to_vn(num // 1000000) + " triệu"
        if num%1000000 <= 10:
            return int_to_vn(num//1000000) + " triệu không nghìn không trăm linh "+int_to_vn(num % 1000000)
        if num % 1000000 < 100:
            return int_to_vn(num // 1000000) + " triệu không nghìn không trăm " + int_to_vn(num % 1000000)
        if num % 1000000 < 1000:
            return int_to_vn(num // 1000000) + " triệu không nghìn " + int_to_vn(num % 1000000)
        if num % 1000000 != 0:
            return int_to_vn(num // 1000000) + " triệu " + int_to_vn(num % 1000000)
    if num // 1000 > 0:
        if num % 1000 == 0: return int_to_vn(num//1000) + " nghìn"
        if num%1000 <=10:
            return int_to_vn(num//1000) + " nghìn không trăm linh "+int_to_vn(num%1000)
        if num%1000 <100:
            return int_to_vn(num//1000) + " nghìn không trăm "+int_to_vn(num%1000)
        if num%1000 != 0:
            return int_to_vn(num//1000) + " nghìn "+int_to_vn(num%1000)
    if num // 100 > 0:
        if num%100 == 0:
            return int_to_vn(num // 100) + " trăm"
        if num%100 <10:
            return int_to_vn(num//100) + " trăm linh " + int_to_vn(num%100)
        if num%100 == 10:
            return int_to_vn(num//100) + " trăm mười"
        if num%100 != 0:
            return int_to_vn(num//100) +  " trăm " + int_to_vn(num%100)
    if num // 10 > 0 and num >= 20:
        if num%10 != 0:
            if num%10 == 5:
                return int_to_vn(num//10) + ' mươi lăm'
            if num%10 == 1:
                return int_to_vn(num//10) + ' mươi mốt'
            if num%10 == 4:
                return int_to_vn(num//10) + ' mươi tư'
            return int_to_vn(num // 10) + ' mươi ' + int_to_vn(num % 10)
        return int_to_vn(num//10) + ' mươi'
    if num // 10 > 0:
        if num == 15:
            return 'mười lăm'
        return "mười "+ d[num%10]

def process_number(text):
    out =''
    tokens = nltk.word_tokenize(text)
    for word in tokens:
        word_out = word
        if word.isdigit():
            word_out = int_to_vn(int(word))
        date = re.findall(r"[\w']+", word)
        if len(date)==3 and date[0].isdigit() and date[1].isdigit() and date[2].isdigit():
            word_out = int_to_vn(int(date[0]))+' tháng '+int_to_vn(int(date[1]))+' năm ' +int_to_vn(int(date[2]))

        if len(date) == 2 and date[0].isdigit() and date[1].isdigit() and int(date[1])>12:
            word_out = 'tháng ' + int_to_vn(int(date[0])) + ' năm ' + int_to_vn(int(date[1]))

        if len(date) == 2 and date[0].isdigit() and date[1].isdigit() and int(date[1])<=12:
            word_out = ' '+int_to_vn(int(date[0])) + ' tháng ' + int_to_vn(int(date[1]))

        if len(date) == 2 and not date[0].isdigit() and not date[1].isdigit():
            word_out = date[0]+ ' trên '+ date[1]

        num = word.split('.')
        if len(num) ==2 and num[0].isdigit() and num[1].isdigit():
            word_out = int_to_vn(int(num[0]+num[1]))

        mnum = word.split(',')
        if len(mnum) ==2 and mnum[0].isdigit() and mnum[1].isdigit():
            word_out = int_to_vn(int(mnum[0]+mnum[1]))

        if word_out == '%': word_out = 'phần trăm'
        if word_out == tokens[0] or word_out == ',' or word_out == ';' or word_out == '.' or tokens[tokens.index(word) -1] == '(' or\
                        word_out == ':' or word_out == '?' or word_out == '!' or word_out == ')' or word_out == '\'':
            out += word_out
        else: out+= ' ' + word_out
    return out

# if __name__=="__main__":
#     print(processNumber('Ngày 2/10/1996 đã có 1000 người tham gia chiến dịch'))
#     print(int_to_vn(1000005))