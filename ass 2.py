import numpy as np
def railfence_encrypt():
        s=str(input("Enter original plain text:"))
        key=int(input("Enter the value of key(depth):"))
        mat=[['\n' for _ in range(len(s))] for _ in range(0,key)]
        st=""
        row,col=0,0
        dir=False
        for i in s:
            if row==0 or row==(key-1):
                dir=not dir
            if i:    
              mat[row][col]=i
            else:
                mat[row][col]='x'  
            col=col+1
            if dir:
                row=row+1
            else:
                row=row-1
        for x in range(0,len(mat)):
            for y in range(0,len(mat[0])):
                if mat[x][y]!='\n':
                  st=st+mat[x][y]
        return st
def railfence_decrypt():
        cipher=str(input("Enter cipher text:"))
        key=int(input("Enter key(depth):"))
        dir=False
        row,col=0,0
        s=""
        mat=[['\n' for _ in range(len(cipher))]for _ in range(key)]
        for _ in range(len(cipher)):
            if row==0 or row==(key-1):
                dir=not dir
            mat[row][col]='*'
            col=col+1
            if dir:
                row=row+1
            else:
                row=row-1
        i=0        
        for x in range(len(mat)):
                for y in range(len(mat[0])):
                    if mat[x][y]=='*':
                        mat[x][y]=cipher[i]
                        i=i+1    
                        
        row,col=0,0
        dir=False                
        for _ in range(len(cipher)):
            if row==0 or row==(key-1):
                dir=not dir
            s=s+mat[row][col]
            col=col+1
            if dir:
                row=row+1
            else:
                row=row-1 
        return s
def row_column_transposition_cipher(encrypt=True):
    text=str(input("Enter plain text to encrypt:"))
    key=str(input("Enter value of key:"))
    text = text.replace(" ", "")
    n = len(key)
    m = int(np.ceil(len(text) / n))
    matrix = [['X' for _ in range(n)] for _ in range(m)]
    
    if encrypt:
        for i, char in enumerate(text):
            matrix[i // n][i % n] = char
        sorted_key = sorted(list(enumerate(key)), key=lambda x: x[1])
        return "".join("".join(row[i[0]] for row in matrix) for i in sorted_key)
    else:
        sorted_key = sorted(list(enumerate(key)), key=lambda x: x[1])
        col_order = [i[0] for i in sorted_key]
        index, text_matrix = 0, [["" for _ in range(n)] for _ in range(m)]
        for col in col_order:
            for row in range(m):
                if index < len(text):
                    text_matrix[row][col] = text[index]
                    index += 1
        return "".join("".join(row) for row in text_matrix).rstrip('X')

print("Railfence:")
c=railfence_encrypt()
print("railfence_Encrypted text:",c)
# p=railfence_decrypt()
# print("railfence_original text:",p)
print("Row column:")
o=row_column_transposition_cipher()
print("Row_column encrypted text:",o)

