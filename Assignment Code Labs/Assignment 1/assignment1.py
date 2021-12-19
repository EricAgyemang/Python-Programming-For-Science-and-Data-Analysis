"""
Problem 1 (5 points):
Compute the area of a triangle.
Formula: (base*height)/2
Complete line 9 and 10
"""
base = height = 5

area_triangle = (base*height)/2
output_string = str(area_triangle)
print("The area of the triangle is " + output_string)


"""
Problem 2 (5 points):
Compute the area of a circle.
Formula: PI*radius*radius
Complete line 24 and 25
"""
import math
PI = math.pi
radius = 10

area_circle = PI*(radius**2)
output_circle = str(area_circle)
output_rad=str(radius)
print ("the area of a circle with a radius of "+output_rad +" is "+output_circle)


"""
Problem 3 (5 points):
Check the memory usage for the target integer.
Complete line 35 to 37
"""
target = 12345678
target_str=str(target)
print("integer: "+target_str)
targetbit=bin(target)
bitnum=len(targetbit)-2
bytenum=bitnum/8
bytenum=int(bytenum)
memory_in_bits =str(bitnum)
memory_in_bytes =str(bytenum) 
print ("memory: "+memory_in_bits+" bits or "+memory_in_bytes+" bytes")


"""
Problem 4 (15 points):
"""
country1 = "the-united-states-of-america"
country2 = "the People's Republic of China"
country3 = "jAPAN"
country4 = "Italian Republic"
country5 = "Great Socialist People's Libyan Arab Jamahiriya"


#Your code goes below:

US1=country1[4]
US1U=US1.upper()
US2=country1[11]
US2U=US2.upper()
US3=country1[21]
US3U=US3.upper()
print(US1U+'.'+US2U+'.'+US3U+'.')

C1=country2[4]
C2=country2[13]
C3=country2[25:30]
print(C1+'.'+C2+'. '+C3)

J1=country3[0]
J1U=J1.upper()
J2=country3[1]
J2L=J2.lower()
J3=country3[2]
J3L=J3.lower()
J4=country3[3]
J4L=J4.lower()
J5=country3[4]
J5L=J5.lower()
print(J1U+J2L+J3L+J4L+J5L)

I1=country4[0:4]
print(I1+'y')

L1=country5[25:30]
print(L1)



"""
Problem 5
"""

#Your code goes below:

x=7

h=0.0000000001

solutioncf= 3*x**2+4*x
solutionnum=(((x+h)**3+2*(x+h)**2+5)-(x**3+2*x**2+5))/h