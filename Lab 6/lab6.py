#!/usr/bin/env python
# coding: utf-8

# In[17]:


f=open("song.txt","r")
g=open("reversed_song.txt","w")
class Reverse:
    def __init__(self,para):
        self.f=para
    def reverse(self):
        for line in self.f:
            g.write(line[::-1])
reverser=Reverse(f)
reverser.reverse()
f.close()
g.close()


# In[32]:


f=open("information.txt","r")
L=[]
D={}
class FileParser:
    def __init__(self,par1):
        self.f=par1
        
    def parse(self,L,D):
        
        for line in self.f:
            line=line.strip()
            words=line.split(' ')
            for i in words:
                if i.isnumeric():
                    L.append(i)
                elif i in D:
                    D[i]=D[i]+1
                else:
                    D[i]=1
parser=FileParser(f)
parser.parse(L,D)
h.close()
print('the list of numbers is:'+str(L))
print('the dictionary is:'+str(D))
                


# In[ ]:




