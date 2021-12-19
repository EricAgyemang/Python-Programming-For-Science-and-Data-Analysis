#!/usr/bin/env python
# coding: utf-8

# In[29]:


class Person(object):
    nPerson=0
    def __init__(self,nameone,nametwo):
        self.nameone=nameone
        self.nametwo=nametwo
    def FName(self):
        print(self.nameone+' '+self.nametwo)
        
class Employee(Person):
    nEmployee=0
    def __init__(self,nameone,nametwo,pay,employID):
        super(Employee,self).__init__(nameone,nametwo)
        self.pay=pay
        self.employID=employID
        Employee.nEmployee +=1
        
class Programmer(Employee):
    def __init__(self,nameone,nametwo,pay,employID,prolang):
        super(Programmer,self).__init__(nameone,nametwo,pay,employID)
        self.prolang=prolang

    
class Manager(Employee):
    def __init__(self,nameone,nametwo,pay,employID,proglist):
        super(Manager,self).__init__(nameone,nametwo,pay,employID)
        self.proglist=[]
    def addProgrammer(self,Programmer):
        self.proglist.append(Programmer)
    def removeProgrammer(self,Programmer):
        self.proglist.remove(Programmer)

a=Programmer('Alice','Arbuncle',35000,12345,'java')
b=Programmer('Bob','Barry',56000,67890,'C++')
c=Programmer('Charlie','Chesterton',45000,24680,'python')
d=Manager('Donna','Doe',95000,13579,[])
m1=d.addProgrammer(a) 
m2=d.addProgrammer(b)
m3=d.addProgrammer(c)
rm=d.removeProgrammer(a)


# In[ ]:




