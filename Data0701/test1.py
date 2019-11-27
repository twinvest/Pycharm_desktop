# lst1 = ['a', 'b', ['qwer', 'xy']]
# lst2 = lst1[:]
# print(id(lst1), id(lst2))
# lst2[2][0] = 'qz'
#
# print(lst1, lst2)

# value_to_print = ''
# for char in 'string_values':
#     value_to_print += '\n' if char == '_' else char
#
# print(value_to_print)

# result1 = []
# result2 = []
# name = input("Please your name : ")
# length = len(name)
#
#
# for i in range(length):
#     if i % 2 == 0:
#         result1.append(name[i])
#     else:
#         result2.append(name[i])
# print(result1)
# print(result2)

#선생님이 짠 코드
name = input("Please your name : ")
odd_seq = ""
even_seq = ""
for x in name[::2]:
    odd_seq += x + '='
for x in name[1::2]:
    even_seq += '='+x
even_seq += "=" * (len(odd_seq) - len(even_seq))
border_line = '-+' * (len(odd_seq)//2)
print('|' + border_line + '|')
print('|' + odd_seq + '|')
print('|' + even_seq + '|')
print('|' + border_line + '|')

#깔끔한 코드
odd_seq = ''.join([x+"=" for x in name[::2]])
even_seq = ''.join(["="+x for x in name[1::2]])
even_seq += "=" * (len(odd_seq) - len(even_seq))
head_footer = '-+' * (len(odd_seq)//2)

print("|"+head_footer+"|")
print("|"+odd_seq+"|")
print("|"+even_seq+"|")
print("|"+head_footer+"|")

#대칭판단
input_list = [2,4,4,2]
result=False
if input_list == input_list[::-1]:
    result = True
print(result)