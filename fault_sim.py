import os
import sys
import argparse
from random import randint
import numpy as np
import math
import kernelgenpt_evaluator as kgp
import time
from multiprocessing import Pool
from functools import partial

tpl_count=0
iso_count=0

#listx_bob=[66, 33, 17, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1,
#2, 1, 1, 16, 8, 4, 2, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 32,
#16, 8, 4, 3, 1, 1, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 16, 8, 4,
#2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1]

listx_bob=[112, 63, 32, 16, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 16, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 31, 16, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 15, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 49, 31, 16, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 15, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 7, 4, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 21, 12, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 9, 5, 3, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1, 1]

#listx_bob=[89, 55, 34, 21, 13, 9, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 13, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 21, 13, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 34, 21, 13, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 13, 8, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 5, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1]
#listx_bob=[121, 60, 31, 15, 7, 3, 1, 2, 1, 4, 2, 1, 2, 1, 1, 8, 4, 2, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 16, 8, 4, 2, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 31, 15, 7, 4, 2, 1, 2, 1, 1, 4, 2, 1, 2, 1, 1, 8, 4, 2, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 16, 8, 4, 2, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 63, 31, 15, 7, 3, 1, 2, 1, 4, 2, 1, 2, 1, 1, 8, 4, 2, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 16, 8, 4, 2, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 32, 16, 8, 4, 2, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 16, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1, 8, 4, 2, 1, 1, 2, 1, 1, 4, 2, 1, 1, 2, 1, 1]

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        return None  # modular inverse does not exist
    else:
        return x % m


p = (2**372)*(3**239)-1
p_2 = p-2
#R=2**(33*23)
#R=2**(16*50)
R = 2**(51*15)
M = p
M_1 = modinv(M,R)
R_inv = modinv(R,M)
m_1 = (R*(R_inv%M)-1)//M
k = 2**15
m_prime = (m_1%k)*M
R_2 = (R*R)%M
global MUL_COUNT
MUL_COUNT = 0
ADD_COUNT = 0

   
Alice_PA_0=0x00004514F8CC94B140F24874F8B87281FA6004CA5B3637C68AC0C0BDB29838051F385FBBCC300BBB24BFBBF6710D7DC8B29ACB81E429BD1BD5629AD0ECAD7C90622F6BB801D0337EE6BC78A7F12FDCB09DECFAE8BFD643C89C3BAC1D87F8B6FA
Alice_PA_1=0x0000158ABF500B5914B3A96CED5FDB37D6DD925F2D6E4F7FEA3CC16E1085754077737EA6F8CC74938D971DA289DCF2435BCAC1897D2627693F9BB167DC01BE34AC494C60B8A0F65A28D7A31EA0D54640653A8099CE5A84E4F0168D818AF02041

Alice_QA_0=0x00001723D2BFA01A78BF4E39E3A333F8A7E0B415A17F208D3419E7591D59D8ABDB7EE6D2B2DFCB21AC29A40F837983C0F057FD041AD93237704F1597D87F074F682961A38B5489D1019924F8A0EF5E4F1B2E64A7BA536E219F5090F76276290E
Alice_QA_1=0x00002569D7EAFB6C60B244EF49E05B5E23F73C4F44169A7E02405E90CEB680CB0756054AC0E3DCE95E2950334262CC973235C2F87D89500BCD465B078BD0DEBDF322A2F86AEDFDCFEE65C09377EFBA0C5384DD837BEDB710209FBC8DDB8C35C7

Alice_RA_0=0x00006066E07F3C0D964E8BC963519FAC8397DF477AEA9A067F3BE343BC53C883AF29CCF008E5A30719A29357A8C33EB3600CD078AF1C40ED5792763A4D213EBDE44CC623195C387E0201E7231C529A15AF5AB743EE9E7C9C37AF3051167525BB
Alice_RA_1=0x000050E30C2C06494249BC4A144EB5F31212BD05A2AF0CB3064C322FC3604FC5F5FE3A08FB3A02B05A48557E15C992254FFC8910B72B8E1328B4893CDCFBFC003878881CE390D909E39F83C5006E0AE979587775443483D13C65B107FADA5165
    
Bob_PB_0  =0x0000605D4697A245C394B98024A5554746DC12FF56D0C6F15D2F48123B6D9C498EEE98E8F7CD6E216E2F1FF7CE0C969CCA29CAA2FAA57174EF985AC0A504260018760E9FDF67467E20C13982FF5B49B8BEAB05F6023AF873F827400E453432FE
Bob_PB_1  =0x0

Bob_QB_0  =0x00005BF9544781803CBD7E0EA8B96D934C5CBCA970F9CC327A0A7E4DAD931EC29BAA8A854B8A9FDE5409AF96C5426FA375D99C68E9AE714172D7F04502D45307FA4839F39A28338BBAFD54A461A535408367D5132E6AA0D3DA6973360F8CD0F1
Bob_QB_1  =0x0

Bob_RB_0  =0x000055E5124A05D4809585F67FE9EA1F02A06CD411F38588BB631BF789C3F98D1C3325843BB53D9B011D8BD1F682C0E4D8A5E723364364E40DAD1B7A476716AC7D1BA705CCDD680BFD4FE4739CC21A9A59ED544B82566BF633E8950186A79FE3
Bob_RB_1  =0x00005AC57EAFD6CC7569E8B53A148721953262C5B404C143380ADCC184B6C21F0CAFE095B7E9C79CA88791F9A72F1B2F3121829B2622515B694A16875ED637F421B539E66F2FEF1CE8DCEFC8AEA608055E9C44077266AB64611BF851BA06C821


mont_one=(R)%M
three_m=3*M
two_m=2*M

def log2(n):
     if n < 0: raise ValueError
     if n==0: return 0
     i = -1
     while n:
         k=n%2
         n = n//2
         i += 1
        
     if k==1:
        i=i+1
     return i

def num_block_gen(a,m):
    r=2**15;
    a_block = []
    for i in range(m):
        a_block.append(0)
    i=0
    while(a>0):
        t1=a%r
        a_block[i]=t1
        i=i+1
        a=a//r
    return a_block

def mont_mul_fp(A,B,m_prime):
    a_mont,b_mont = A,B
    a_block = num_block_gen(a_mont,54)
    S = []
    for i in range(55):
        S.append(0)
    k=2**15
    for i in range(54):
        q = S[i]%k
        S[i+1] = (S[i]+q*m_prime)//k + a_block[i]*b_mont
    return S[52]

def mont_mul_fp2(A_0,A_1,B_0,B_1,m_prime):
    t00 = mont_mul_fp(A_0,B_0,m_prime)
    t11 = mont_mul_fp(A_1,B_1,m_prime)
    t01 = mont_mul_fp(A_0,B_1,m_prime)
    t10 = mont_mul_fp(A_1,B_0,m_prime)
    c_0 = (t00-t11)%(2*m_prime)
    c_1 = (t01+t10)%(2*m_prime)
    return (c_0,c_1)

def mont_add_fp2(A_0,A_1,B_0,B_1,m_prime):
    c_0 = (A_0+B_0)%(2*m_prime)
    c_1 = (A_1+B_1)%(2*m_prime)
    global ADD_COUNT
    ADD_COUNT = ADD_COUNT+1
    return (c_0,c_1)

def mont_sub_fp2(A_0,A_1,B_0,B_1,m_prime):
    c_0 = (A_0-B_0)%(2*m_prime)
    c_1 = (A_1-B_1)%(2*m_prime)
    global ADD_COUNT
    ADD_COUNT = ADD_COUNT+1
    return (c_0,c_1)

def mont_mul_karat(A_0,A_1,B_0,B_1,m_prime):
    add_1 = (A_0+A_1)%(2*m_prime)
    add_2 = (B_0-B_1)%(2*m_prime)
    mul1 = mont_mul_fp(A_0,B_1,m_prime)
    mul2 = mont_mul_fp(A_1,B_0,m_prime)
    mul3 = mont_mul_fp(add_1,add_2,m_prime)
    c_1 = (mul1+mul2)%(2*m_prime)
    c_0 = (mul3+(mul1-mul2))%(2*m_prime)
    global MUL_COUNT
    MUL_COUNT = MUL_COUNT+1

    return (c_0,c_1)

def mont_mul_sq(A_0,A_1,m_prime):
    mul1 = mont_mul_fp(A_0,A_0,m_prime)
    mul2 = mont_mul_fp(A_1,A_1,m_prime)
    mul3 = mont_mul_fp(A_0,A_1,m_prime)
    c_0 = (mul1-mul2)%(2*m_prime)
    c_1 = (mul3+mul3)%(2*m_prime)
    return (c_0,c_1)

def convert_to_bin(c_hb,c_h):
    temp_1 = c_h
    i = 0
    while(temp_1>0):
        c_hb[i] = temp_1%2    
        temp_1 = temp_1//2
        i = i+1
    return

def inv_fp(x):
     p_2_bin = np.zeros(752,dtype=int)
     convert_to_bin(p_2_bin,p_2)
     temp = x
     Z2_inv_mont = mont_one
     inv_fp_count = 0
     for i in range(0,752,1):
        inv_fp_count = inv_fp_count+1
        if(p_2_bin[i]==1):
            Z2_inv_mont = mont_mul_fp(Z2_inv_mont,temp,m_prime)
        temp = mont_mul_fp(temp,temp,m_prime)
     temp = mont_mul_fp(Z2_inv_mont,x,m_prime)
     temp = mont_mul_fp(temp,1,m_prime)     
     return Z2_inv_mont

def inv_fp2(x_0,x_1):
     t1_0 = mont_mul_fp(x_0,x_0,m_prime)
     t1_1 = mont_mul_fp(x_1,x_1,m_prime)
     t1_0 = (t1_0+t1_1)%(2*m_prime)
     t1_0 = inv_fp(t1_0)     
     x_1 = (0-x_1)%(2*m_prime)
     x_0 = mont_mul_fp(x_0,t1_0,m_prime)     
     x_1 = mont_mul_fp(x_1,t1_0,m_prime)
     return x_0,x_1

def precompute(X_0,X_1,A_0,A_1,m_prime):
     list_double = list()
     list_double.append(X_0)
     list_double.append(X_1)
     mont_one = (R)%M
     for i in range(378):
          Z_0 = mont_one
          Z_1 = 0
          t0_0,t0_1 = mont_sub_fp2(X_0,X_1,Z_0,Z_1,m_prime)
          t1_0,t1_1 = mont_add_fp2(X_0,X_1,Z_0,Z_1,m_prime)
          t0_0,t0_1 = mont_mul_sq(t0_0,t0_1,m_prime)
          t1_0,t1_1 = mont_mul_sq(t1_0,t1_1,m_prime)
          X2P_0,X2P_1 = mont_mul_karat(t0_0,t0_1,t1_0,t1_1,m_prime)
          t1_0,t1_1 = mont_sub_fp2(t1_0,t1_1,t0_0,t0_1,m_prime)
          t3_0,t3_1 = mont_mul_karat(A_0,A_1,t1_0,t1_1,m_prime)
          Z2P_0,Z2P_1 = mont_add_fp2(t0_0,t0_1,t3_0,t3_1,m_prime)
          Z2P_0,Z2P_1 = mont_mul_karat(Z2P_0,Z2P_1,t1_0,t1_1,m_prime)
          temp_0,temp_1 = inv_fp2(Z2P_0,Z2P_1)
          x,unused = mont_mul_karat(X2P_0,X2P_1,temp_0,temp_1,m_prime)
          X_0 = x
          X_1 = unused
          list_double.append(x)
          list_double.append(unused)
     print("precomputation done")
     return list_double
     
def precomputeadd(X_P_0,X_P_1,X_Q_0,X_Q_1,Z_Q_0,Z_Q_1,X_D_0,X_D_1,Z_D_0,Z_D_1,m_prime):
     t0_0,t0_1 = mont_add_fp2(Z_D_0,Z_D_1,Z_D_0,Z_D_1,m_prime)
     t1_0,t1_1 = mont_add_fp2(t0_0,t0_1,t0_0,t0_1,m_prime)
     t2_0,t2_1 = mont_add_fp2(X_D_0,X_D_1,X_D_0,X_D_1,m_prime)
     t3_0,t3_1 = mont_add_fp2(t2_0,t2_1,t2_0,t2_1,m_prime)
     t4_0,t4_1 = mont_mul_karat(X_P_0,X_P_1,X_Q_0,X_Q_1,m_prime)
     t5_0,t5_1 = mont_mul_karat(X_P_0,X_P_1,Z_Q_0,Z_Q_1,m_prime)
     t6_0,t6_1 = mont_sub_fp2(t4_0,t4_1,Z_Q_0,Z_Q_1,m_prime)
     t7_0,t7_1 = mont_sub_fp2(t5_0,t5_1,X_Q_0,X_Q_1,m_prime)
     t8_0,t8_1 = mont_mul_sq(t6_0,t6_1,m_prime)
     t9_0,t9_1 = mont_mul_sq(t7_0,t7_1,m_prime)
     X_PQ_0,X_PQ_1 = mont_mul_karat(t1_0,t1_1,t8_0,t8_1,m_prime)
     Z_PQ_0,Z_PQ_1 = mont_mul_karat(t3_0,t3_1,t9_0,t9_1,m_prime)
     Z_P_0,Z_P_1 = mont_one,0
     t0_0,t0_1 = mont_sub_fp2(X_P_0,X_P_1,Z_P_0,Z_P_1,m_prime)
     t1_0,t1_1 = mont_add_fp2(X_P_0,X_P_1,Z_P_0,Z_P_1,m_prime)
     t2_0,t2_1 = mont_add_fp2(X_Q_0,X_Q_1,Z_Q_0,Z_Q_1,m_prime)
     t3_0,t3_1 = mont_sub_fp2(X_Q_0,X_Q_1,Z_Q_0,Z_Q_1,m_prime)
     t4_0,t4_1 = mont_mul_karat(t0_0,t0_1,t2_0,t2_1,m_prime)
     t5_0,t5_1 = mont_mul_karat(t1_0,t1_1,t3_0,t3_1,m_prime)
     t6_0,t6_1 = mont_add_fp2(t4_0,t4_1,t5_0,t5_1,m_prime)
     t7_0,t7_1 = mont_sub_fp2(t4_0,t4_1,t5_0,t5_1,m_prime)
     t8_0,t8_1 = mont_mul_sq(t6_0,t6_1,m_prime)
     t9_0,t9_1 = mont_mul_sq(t7_0,t7_1,m_prime)
     CX_PQ_0,CX_PQ_1=mont_mul_karat(Z_D_0,Z_D_1,t8_0,t8_1,m_prime)
     CZ_PQ_0,CZ_PQ_1=mont_mul_karat(X_D_0,X_D_1,t9_0,t9_1,m_prime)
     if(X_PQ_1!=CX_PQ_1):
     	print("error")
     return X_PQ_0, X_PQ_1,Z_PQ_0,Z_PQ_1

def dbladd(X_P_0,X_P_1,Z_P_0,Z_P_1,X_Q_0,X_Q_1,Z_Q_0,Z_Q_1,X_D_0,X_D_1,Z_D_0,Z_D_1,A24plus_0,A24plus_1, m_prime):
     (t0_0,t0_1) = mont_add_fp2(X_P_0,X_P_1,Z_P_0,Z_P_1,m_prime)
     (t1_0,t1_1) = mont_sub_fp2(X_P_0,X_P_1,Z_P_0,Z_P_1,m_prime)
     (X2P_0,X2P_1) = mont_mul_sq(t0_0,t0_1,m_prime)
     (t2_0,t2_1) = mont_sub_fp2(X_Q_0,X_Q_1,Z_Q_0,Z_Q_1,m_prime)
     (X_PQ_0,X_PQ_1) = mont_add_fp2(X_Q_0,X_Q_1,Z_Q_0,Z_Q_1,m_prime)
     (t0_0,t0_1) = mont_mul_karat(t0_0,t0_1,t2_0,t2_1,m_prime)
     (Z2P_0,Z2P_1) = mont_mul_sq(t1_0,t1_1,m_prime)
     (t1_0,t1_1) = mont_mul_karat(t1_0,t1_1,X_PQ_0,X_PQ_1,m_prime)
     (t2_0,t2_1) = mont_sub_fp2(X2P_0,X2P_1,Z2P_0,Z2P_1,m_prime)
     (X2P_0,X2P_1) = mont_mul_karat(X2P_0,X2P_1,Z2P_0,Z2P_1,m_prime)
     (X_PQ_0,X_PQ_1) = mont_mul_karat(A24plus_0,A24plus_1,t2_0,t2_1,m_prime)
     (Z_PQ_0,Z_PQ_1) = mont_sub_fp2(t0_0,t0_1,t1_0,t1_1,m_prime)
     (Z2P_0,Z2P_1) = mont_add_fp2(X_PQ_0,X_PQ_1,Z2P_0,Z2P_1,m_prime)
     (X_PQ_0,X_PQ_1) = mont_add_fp2(t0_0,t0_1,t1_0,t1_1,m_prime)
     (Z2P_0,Z2P_1) = mont_mul_karat(t2_0,t2_1,Z2P_0,Z2P_1,m_prime)
     (Z_PQ_0,Z_PQ_1) = mont_mul_sq(Z_PQ_0,Z_PQ_1,m_prime)
     (X_PQ_0,X_PQ_1) = mont_mul_sq(X_PQ_0,X_PQ_1,m_prime)
     (Z_PQ_0,Z_PQ_1) = mont_mul_karat(X_D_0,X_D_1,Z_PQ_0,Z_PQ_1,m_prime)
     (X_PQ_0,X_PQ_1) = mont_mul_karat(Z_D_0,Z_D_1,X_PQ_0,X_PQ_1,m_prime)
     return X2P_0,X2P_1,Z2P_0,Z2P_1,X_PQ_0,X_PQ_1,Z_PQ_0,Z_PQ_1

def swap(R_X_0,R_X_1,R2_X_0,R2_X_1,R_Z_0,R_Z_1,R2_Z_0,R2_Z_1,choice):
    if choice==1:
        temp = R_X_0
        R_X_0 = R2_X_0
        R2_X_0 = temp
        temp = R_X_1
        R_X_1 = R2_X_1
        R2_X_1 = temp
        temp = R_Z_0
        R_Z_0 = R2_Z_0
        R2_Z_0 = temp
        temp = R_Z_1
        R_Z_1 = R2_Z_1
        R2_Z_1 = temp
    return (R_X_0,R_X_1,R2_X_0,R2_X_1,R_Z_0,R_Z_1,R2_Z_0,R2_Z_1)

def kernelgen(X_P_0,X_P_1,Z_P_0,Z_P_1,X_Q_0,X_Q_1,Z_Q_0,Z_Q_1,X_D_0,X_D_1,Z_D_0,Z_D_1,key,A_0,A_1,m_prime):
     key_temp = key
     key_bin = np.zeros(379)
     convert_to_bin(key_bin,key_temp)
     prev_bit = 0
     for j in range(378):
          s_i_1 = key_bin[j]
          s_i = prev_bit
          c = int(s_i) ^ int(s_i_1)
          prev_bit=key_bin[j]
          X_P_0,X_P_1,X_D_0,X_D_1,Z_P_0,Z_P_1,Z_D_0,Z_D_1=swap(X_P_0,X_P_1,X_D_0,X_D_1,Z_P_0,Z_P_1,Z_D_0,Z_D_1,c)
          (X_Q_0,X_Q_1,Z_Q_0,Z_Q_1,X_D_0,X_D_1,Z_D_0,Z_D_1)=dbladd(X_Q_0,X_Q_1,Z_Q_0,Z_Q_1,X_D_0,X_D_1,Z_D_0,Z_D_1,X_P_0,X_P_1,Z_P_0,Z_P_1,A_0,A_1,m_prime)
     c=int(0) ^ int(key_bin[377])     
     X_P_0,X_P_1,X_D_0,X_D_1,Z_P_0,Z_P_1,Z_D_0,Z_D_1=swap(X_P_0,X_P_1,X_D_0,X_D_1,Z_P_0,Z_P_1,Z_D_0,Z_D_1,c)
     return X_P_0,X_P_1,Z_P_0,Z_P_1

def precomputekernelgen(X_P_0,X_P_1,Z_P_0,Z_P_1,X_Q_0,X_Q_1,X_D_0,X_D_1,Z_D_0,Z_D_1,A_0,A_1,key,m_prime):
     key_temp = key
     key_bin = np.zeros(379)
     convert_to_bin(key_bin,key_temp)
     list_Q=precompute(X_Q_0,X_Q_1,A_0,A_1,m_prime)
     for j in range(378):
          i = 377-j
          s_i_1 = key_bin[i+1]
          s_i = key_bin[i]
          c = int(s_i) ^ int(s_i_1)
          X_P_0,X_P_1,X_D_0,X_D_1,Z_P_0,Z_P_1,Z_D_0,Z_D_1 = swap(X_P_0,X_P_1,X_D_0,X_D_1,Z_P_0,Z_P_1,Z_D_0,Z_D_1,c)
          (X_D_0,X_D_1,Z_D_0,Z_D_1) = precomputeadd(list_Q[2*j],list_Q[2*j+1],X_D_0,X_D_1,Z_D_0,Z_D_1,X_P_0,X_P_1,Z_P_0,Z_P_1,m_prime)     
     c = int(0) ^ int(key_bin[0])     
     X_P_0,X_P_1,X_D_0,X_D_1,Z_P_0,Z_P_1,Z_D_0,Z_D_1 = swap(X_P_0,X_P_1,X_D_0,X_D_1,Z_P_0,Z_P_1,Z_D_0,Z_D_1,c)
     return X_P_0,X_P_1,Z_P_0,Z_P_1

def xDBL(X_0,X_1,Z_0,Z_1,A24plus_0,A24plus_1,C24_0,C24_1,flag,double_total_cost):
     (t0_0,t0_1) = mont_sub_fp2(X_0,X_1,Z_0,Z_1,m_prime)
     (t1_0,t1_1) = mont_add_fp2(X_0,X_1,Z_0,Z_1,m_prime)
     (t2_0,t2_1) = mont_mul_sq(t0_0,t0_1,m_prime)
     (t3_0,t3_1) = mont_mul_sq(t1_0,t1_1,m_prime)
     (t4_0,t4_1) = mont_mul_karat(t2_0,t2_1,C24_0,C24_1,m_prime)
     (CX_0,CX_1) = mont_mul_karat(t4_0,t4_1,t3_0,t3_1,m_prime)
     (t6_0,t6_1) = mont_sub_fp2(t3_0,t3_1,t2_0,t2_1,m_prime)
     (t7_0,t7_1) = mont_mul_karat(t6_0,t6_1,A24plus_0,A24plus_1,m_prime)
     (t8_0,t8_1) = mont_add_fp2(t4_0,t4_1,t7_0,t7_1,m_prime)
     (CZ_0,CZ_1) = mont_mul_karat(t8_0,t8_1,t6_0,t6_1,m_prime)
     (t0_0,t0_1) = mont_sub_fp2(X_0,X_1,Z_0,Z_1,m_prime)
     (t1_0,t1_1) = mont_add_fp2(X_0,X_1,Z_0,Z_1,m_prime)
     (t0_0,t0_1) = mont_mul_sq(t0_0,t0_1,m_prime)
     (t1_0,t1_1) = mont_mul_sq(t1_0,t1_1,m_prime)
     (Z_0,Z_1) = mont_mul_karat(C24_0,C24_1,t0_0,t0_1,m_prime)
     (X_0,X_1) = mont_mul_karat(Z_0,Z_1,t1_0,t1_1,m_prime)
     (t1_0,t1_1) = mont_sub_fp2(t1_0,t1_1,t0_0,t0_1,m_prime)
     (t0_0,t0_1) = mont_mul_karat(A24plus_0,A24plus_1,t1_0,t1_1,m_prime)
     (Z_0,Z_1) = mont_add_fp2(Z_0,Z_1,t0_0,t0_1,m_prime)
     (Z_0,Z_1) = mont_mul_karat(Z_0,Z_1,t1_0,t1_1,m_prime)
     double_total_cost = double_total_cost+1
     if((CX_0!=X_0) or (CX_1!=X_1) or (CZ_0!=Z_0) or (CZ_1!=Z_1)):
          print("Fail Fail")
     return X_0,X_1,Z_0,Z_1,double_total_cost


def xDBLe(R_X_0,R_X_1,R_Z_0,R_Z_1,A24plus_0,A24plus_1,C24_0,C24_1,m,flag,double_total_cost):
    X_0=R_X_0
    X_1=R_X_1
    Z_0=R_Z_0
    Z_1=R_Z_1
    for i in range(0,m):
      (X_0,X_1,Z_0,Z_1,double_total_cost)=xDBL(X_0,X_1,Z_0,Z_1,A24plus_0,A24plus_1,C24_0,C24_1,0,double_total_cost)
    return (X_0,X_1,Z_0,Z_1,double_total_cost)

def xTPL(X_0,X_1,Z_0,Z_1,A24plus_0,A24plus_1,A24minus_0,A24minus_1):
    global tpl_count
    tpl_count = tpl_count+1
    (t0_0,t0_1) = mont_sub_fp2(X_0,X_1,Z_0,Z_1,m_prime)
    (t1_0,t1_1) = mont_add_fp2(X_0,X_1,Z_0,Z_1,m_prime)
    (t2_0,t2_1) = mont_mul_sq(t0_0,t0_1,m_prime) 
    (t3_0,t3_1) = mont_mul_sq(t1_0,t1_1,m_prime)
    (t4_0,t4_1) = mont_add_fp2(t1_0,t1_1,t0_0,t0_1,m_prime)#t4:=t1+t0;
    (t5_0,t5_1) = mont_sub_fp2(t1_0,t1_1,t0_0,t0_1,m_prime)#t0:=t1-t0;
    (t7_0,t7_1) = mont_add_fp2(t2_0,t2_1,t3_0,t3_1,m_prime)#t1:=t1-t3
    (t6_0,t6_1) = mont_mul_sq(t4_0,t4_1,m_prime)#t1:=t4^2
    (t9_0,t9_1) = mont_mul_karat(t3_0,t3_1,A24plus_0,A24plus_1,m_prime)#t5:=t3*A24plus
    (t8_0,t8_1) = mont_sub_fp2(t6_0,t6_1,t7_0,t7_1,m_prime)#t1:=t1-t2
    (t10_0,t10_1) = mont_mul_karat(t3_0,t3_1,t9_0,t9_1,m_prime)#t3:=t3*t5
    (t11_0,t11_1) = mont_mul_karat(t2_0,t2_1,A24minus_0,A24minus_1,m_prime)#t6:=t2*A24minus
    (t14_0,t14_1) = mont_sub_fp2(t9_0,t9_1,t11_0,t11_1,m_prime)#t3:=t2-t3
    (t12_0,t12_1) = mont_mul_karat(t2_0,t2_1,t11_0,t11_1,m_prime)#t2:=t2*t6
    (t15_0,t15_1) = mont_mul_karat(t14_0,t14_1,t8_0,t8_1,m_prime)#t2:=t2*t6
    (t13_0,t13_1) = mont_sub_fp2(t12_0,t12_1,t10_0,t10_1,m_prime)#t3:=t2-t3
    (t18_0,t18_1) = mont_sub_fp2(t13_0,t13_1,t15_0,t15_1,m_prime)#t1:=t3-t1
    (t16_0,t16_1) = mont_add_fp2(t13_0,t13_1,t15_0,t15_1,m_prime)#t2:=t3+t1
    (t17_0,t17_1) = mont_mul_sq(t16_0,t16_1,m_prime)#t2:=t2^2
    (t19_0,t19_1) = mont_mul_sq(t18_0,t18_1,m_prime)#t1:=t1^2
    (CX_0,CX_1) = mont_mul_karat(t17_0,t17_1,t4_0,t4_1,m_prime)#X:=t2*t4
    (CZ_0,CZ_1) = mont_mul_karat(t19_0,t19_1,t5_0,t5_1,m_prime)#X:=t2*t4     
    (t0_0,t0_1) = mont_sub_fp2(X_0,X_1,Z_0,Z_1,m_prime)           
    (t2_0,t2_1) = mont_mul_sq(t0_0,t0_1,m_prime) 
    (t1_0,t1_1) = mont_add_fp2(X_0,X_1,Z_0,Z_1,m_prime) 
    (t3_0,t3_1) = mont_mul_sq(t1_0,t1_1,m_prime)
    (t4_0,t4_1) = mont_add_fp2(t1_0,t1_1,t0_0,t0_1,m_prime)#t4:=t1+t0;
    (t0_0,t0_1) = mont_sub_fp2(t1_0,t1_1,t0_0,t0_1,m_prime)#t0:=t1-t0;
    (t1_0,t1_1) = mont_mul_sq(t4_0,t4_1,m_prime)#t1:=t4^2
    (t1_0,t1_1) = mont_sub_fp2(t1_0,t1_1,t3_0,t3_1,m_prime)#t1:=t1-t3
    (t1_0,t1_1) = mont_sub_fp2(t1_0,t1_1,t2_0,t2_1,m_prime)#t1:=t1-t2
    (t5_0,t5_1) = mont_mul_karat(t3_0,t3_1,A24plus_0,A24plus_1,m_prime)#t5:=t3*A24plus
    (t3_0,t3_1) = mont_mul_karat(t3_0,t3_1,t5_0,t5_1,m_prime)#t3:=t3*t5
    (t6_0,t6_1) = mont_mul_karat(t2_0,t2_1,A24minus_0,A24minus_1,m_prime)#t6:=t2*A24minus
    (t2_0,t2_1) = mont_mul_karat(t2_0,t2_1,t6_0,t6_1,m_prime)#t2:=t2*t6
    (t3_0,t3_1) = mont_sub_fp2(t2_0,t2_1,t3_0,t3_1,m_prime)#t3:=t2-t3
    (t2_0,t2_1) = mont_sub_fp2(t5_0,t5_1,t6_0,t6_1,m_prime)#t3:=t2-t3
    (t1_0,t1_1) = mont_mul_karat(t2_0,t2_1,t1_0,t1_1,m_prime)#t2:=t2*t6
    (t2_0,t2_1) = mont_add_fp2(t3_0,t3_1,t1_0,t1_1,m_prime)#t2:=t3+t1
    (t2_0,t2_1) = mont_mul_sq(t2_0,t2_1,m_prime)#t2:=t2^2
    (X_0,X_1) = mont_mul_karat(t2_0,t2_1,t4_0,t4_1,m_prime)#X:=t2*t4
    (t1_0,t1_1) = mont_sub_fp2(t3_0,t3_1,t1_0,t1_1,m_prime)#t1:=t3-t1
    (t1_0,t1_1) = mont_mul_sq(t1_0,t1_1,m_prime)#t1:=t1^2
    (Z_0,Z_1) = mont_mul_karat(t1_0,t1_1,t0_0,t0_1,m_prime)#X:=t2*t4
    if (CX_0-X_0!=0 or CX_1-X_1!=0 or CZ_0-Z_0!=0 or CZ_1-Z_1!=0):
         print("error")
         exit()
    return X_0,X_1,Z_0,Z_1

def get_3_iso(X_0,X_1,Z_0,Z_1):                
    (K1_0,K1_1) = mont_sub_fp2(X_0,X_1,Z_0,Z_1,m_prime) # K1=X-Z
    (t0_0,t0_1) = mont_mul_sq(K1_0,K1_1,m_prime) #t0=K1^2
    (K2_0,K2_1) = mont_add_fp2(X_0,X_1,Z_0,Z_1,m_prime) # K2=X+Z
    (t1_0,t1_1) = mont_mul_sq(K2_0,K2_1,m_prime) #t1=K2^2
    (t2_0,t2_1) = mont_add_fp2(t0_0,t0_1,t1_0,t1_1,m_prime) # t2=t0+t1
    (t3_0,t3_1) = mont_add_fp2(K1_0,K1_1,K2_0,K2_1,m_prime) # t3=K1+K2
    (t3_0,t3_1) = mont_mul_sq(t3_0,t3_1,m_prime) #t3=t3^2
    (t3_0,t3_1) = mont_sub_fp2(t3_0,t3_1,t2_0,t2_1,m_prime) # t3=t3-t2
    (t2_0,t2_1) = mont_add_fp2(t3_0,t3_1,t1_0,t1_1,m_prime) # t2=t3+t1
    (t3_0,t3_1) = mont_add_fp2(t0_0,t0_1,t3_0,t3_1,m_prime) # t3=t0+t3
    (t4_0,t4_1) = mont_add_fp2(t0_0,t0_1,t3_0,t3_1,m_prime) # t4=t0+t3
    (t4_0,t4_1) = mont_add_fp2(t4_0,t4_1,t4_0,t4_1,m_prime) # t4=t4+t4
    (t4_0,t4_1) = mont_add_fp2(t4_0,t4_1,t1_0,t1_1,m_prime) # t4=t4+t1
    (A24minus_0,A24minus_1) = mont_mul_karat(t2_0,t2_1,t4_0,t4_1,m_prime)#A24minus=t2.t4
    (t4_0,t4_1) = mont_add_fp2(t1_0,t1_1,t2_0,t2_1,m_prime) # t4=t2+t1
    (t4_0,t4_1) = mont_add_fp2(t4_0,t4_1,t4_0,t4_1,m_prime) # t4=t4+t4
    (t4_0,t4_1) = mont_add_fp2(t0_0,t0_1,t4_0,t4_1,m_prime) # t4=t0+t4
    (t4_0,t4_1) = mont_mul_karat(t3_0,t3_1,t4_0,t4_1,m_prime)#t4=t3.t4
    (t0_0,t0_1) = mont_sub_fp2(t4_0,t4_1,A24minus_0,A24minus_1,m_prime) # t0=t4-A24minus
    (A24plus_0,A24plus_1) = mont_add_fp2(t0_0,t0_1,A24minus_0,A24minus_1,m_prime) # A24plus=A24minus+t0
    return A24plus_0,A24plus_1,A24minus_0,A24minus_1,K1_0,K1_1,K2_0,K2_1

def get_4_iso(X_0,X_1,Z_0,Z_1):
     (K2_0,K2_1) = mont_sub_fp2(X_0,X_1,Z_0,Z_1,m_prime)
     (K3_0,K3_1) = mont_add_fp2(X_0,X_1,Z_0,Z_1,m_prime)
     (K1_0,K1_1) = mont_mul_sq(Z_0,Z_1,m_prime)
     (K1_0,K1_1) = mont_add_fp2(K1_0,K1_1,K1_0,K1_1,m_prime)
     (C24_0,C24_1) = mont_mul_sq(K1_0,K1_1,m_prime)
     (K1_0,K1_1) = mont_add_fp2(K1_0,K1_1,K1_0,K1_1,m_prime)
     (A24plus_0,A24plus_1) = mont_mul_sq(X_0,X_1,m_prime)
     (A24plus_0,A24plus_1) = mont_add_fp2(A24plus_0,A24plus_1,A24plus_0,A24plus_1,m_prime)
     (A24plus_0,A24plus_1) = mont_mul_sq(A24plus_0,A24plus_1,m_prime)
     return A24plus_0,A24plus_1,C24_0,C24_1,K1_0,K1_1,K2_0,K2_1,K3_0,K3_1

def eval_4_iso(X_0,X_1,Z_0,Z_1,K1_0,K1_1,K2_0,K2_1,K3_0,K3_1):
     (t0_0,t0_1) = mont_add_fp2(X_0,X_1,Z_0,Z_1,m_prime)
     (t1_0,t1_1) = mont_sub_fp2(X_0,X_1,Z_0,Z_1,m_prime)
     (t2_0,t2_1) = mont_mul_karat(t0_0,t0_1,K2_0,K2_1,m_prime)
     (t3_0,t3_1) = mont_mul_karat(t1_0,t1_1,K3_0,K3_1,m_prime)
     (t4_0,t4_1) = mont_mul_karat(t0_0,t0_1,t1_0,t1_1,m_prime)
     (t5_0,t5_1) = mont_mul_karat(t4_0,t4_1,K1_0,K1_1,m_prime)
     (t6_0,t6_1) = mont_add_fp2(t2_0,t2_1,t3_0,t3_1,m_prime)
     (t7_0,t7_1) = mont_sub_fp2(t2_0,t2_1,t3_0,t3_1,m_prime)
     (t8_0,t8_1) = mont_mul_sq(t6_0,t6_1,m_prime)
     (t9_0,t9_1) = mont_mul_sq(t7_0,t7_1,m_prime)
     (t10_0,t10_1) = mont_add_fp2(t5_0,t5_1,t8_0,t8_1,m_prime)
     (t11_0,t11_1) = mont_sub_fp2(t9_0,t9_1,t5_0,t5_1,m_prime)
     (CX_0,CX_1) = mont_mul_karat(t10_0,t10_1,t8_0,t8_1,m_prime)
     (CZ_0,CZ_1) = mont_mul_karat(t9_0,t9_1,t11_0,t11_1,m_prime)
     (t0_0,t0_1) = mont_add_fp2(X_0,X_1,Z_0,Z_1,m_prime)
     (t1_0,t1_1) = mont_sub_fp2(X_0,X_1,Z_0,Z_1,m_prime)     
     (X_0,X_1) = mont_mul_karat(t0_0,t0_1,K2_0,K2_1,m_prime)
     (Z_0,Z_1) = mont_mul_karat(t1_0,t1_1,K3_0,K3_1,m_prime)
     (t0_0,t0_1) = mont_mul_karat(t0_0,t0_1,t1_0,t1_1,m_prime)
     (t0_0,t0_1) = mont_mul_karat(t0_0,t0_1,K1_0,K1_1,m_prime)
     (t1_0,t1_1) = mont_add_fp2(X_0,X_1,Z_0,Z_1,m_prime)
     (Z_0,Z_1) = mont_sub_fp2(X_0,X_1,Z_0,Z_1,m_prime)
     (t1_0,t1_1) = mont_mul_sq(t1_0,t1_1,m_prime)
     (Z_0,Z_1) = mont_mul_sq(Z_0,Z_1,m_prime)
     (X_0,X_1) = mont_add_fp2(t1_0,t1_1,t0_0,t0_1,m_prime)
     (t0_0,t0_1) = mont_sub_fp2(Z_0,Z_1,t0_0,t0_1,m_prime)
     (X_0,X_1) = mont_mul_karat(X_0,X_1,t1_0,t1_1,m_prime)
     (Z_0,Z_1) = mont_mul_karat(Z_0,Z_1,t0_0,t0_1,m_prime)
     if((CX_0!=X_0) or (CX_1!=X_1) or (CZ_0!=Z_0) or (CZ_1!=Z_1)):
          print("Fail Fail")
     return X_0,X_1,Z_0,Z_1

def eval_3_iso(X_0,X_1,Z_0,Z_1,K1_0,K1_1,K2_0,K2_1):
    global iso_count
    iso_count = iso_count+1
    (t0_0,t0_1) = mont_add_fp2(X_0,X_1,Z_0,Z_1,m_prime)       
    (t1_0,t1_1) = mont_sub_fp2(X_0,X_1,Z_0,Z_1,m_prime) #t1:=Z3*X;       
    (t0_0,t0_1) = mont_mul_karat(t0_0,t0_1,K1_0,K1_1,m_prime) #t2:=Z3*Z;
    (t1_0,t1_1) = mont_mul_karat(t1_0,t1_1,K2_0,K2_1,m_prime)              
    (t2_0,t2_1) = mont_add_fp2(t0_0,t0_1,t1_0,t1_1,m_prime) #t2:=Z*X3;               
    (t0_0,t0_1) = mont_sub_fp2(t1_0,t1_1,t0_0,t0_1,m_prime) #t1:=t1-t2;
    (t2_0,t2_1) = mont_mul_sq(t2_0,t2_1,m_prime)        
    (t0_0,t0_1) = mont_mul_sq(t0_0,t0_1,m_prime)       
    (X_0,X_1) = mont_mul_karat(X_0,X_1,t2_0,t2_1,m_prime) #X:=X*t0;        
    (Z_0,Z_1) = mont_mul_karat(Z_0,Z_1,t0_0,t0_1,m_prime) #Z:=Z*t1;        
    return X_0,X_1,Z_0,Z_1;

def xTPLe(R_X_0,R_X_1,R_Z_0,R_Z_1,A24plus_0,A24plus_1,A24minus_0,A24minus_1,m):
    X_0,X_1,Z_0,Z_1 = R_X_0,R_X_1,R_Z_0,R_Z_1
    for i in range(0,m):
      (X_0,X_1,Z_0,Z_1) = xTPL(X_0,X_1,Z_0,Z_1,A24plus_0,A24plus_1,A24minus_0,A24minus_1)
    return (X_0,X_1,Z_0,Z_1)

def inv_3_way(PZ_0,PZ_1,QZ_0,QZ_1,RZ_0,RZ_1):
     t0_0,t0_1 = mont_mul_karat(PZ_0,PZ_1,QZ_0,QZ_1,m_prime)
     t1_0,t1_1 = mont_mul_karat(RZ_0,RZ_1,t0_0,t0_1,m_prime)
     t1_0,t1_1 = inv_fp2(t1_0,t1_1)  
     t2_0,t2_1 = mont_mul_karat(RZ_0,RZ_1,t1_0,t1_1,m_prime)
     t3_0,t3_1 = mont_mul_karat(t2_0,t2_1,QZ_0,QZ_1,m_prime)
     QZ_0,QZ_1 = mont_mul_karat(t2_0,t2_1,PZ_0,PZ_1,m_prime)
     RZ_0,RZ_1 = mont_mul_karat(t0_0,t0_1,t1_0,t1_1,m_prime)
     PZ_0,PZ_1 = t3_0,t3_1
     return PZ_0,PZ_1,QZ_0,QZ_1,RZ_0,RZ_1
         
def isogen3(key,Alice_PA_0,Alice_PA_1,Alice_QA_0,Alice_QA_1,Alice_RA_0,Alice_RA_1,Bob_PB_0,Bob_PB_1,Bob_QB_0,Bob_QB_1,Bob_RB_0,Bob_RB_1,abort_row):
   Alice_PA_0,Alice_PA_1 = (Alice_PA_0*R)%M,(Alice_PA_1*R)%M
   Alice_QA_0,Alice_QA_1 = (Alice_QA_0*R)%M,(Alice_QA_1*R)%M
   Alice_RA_0,Alice_RA_1 = (Alice_RA_0*R)%M,(Alice_RA_1*R)%M
   Bob_PB_0,Bob_PB_1= (Bob_PB_0*R)%M,(Bob_PB_1*R)%M
   Bob_QB_0,Bob_QB_1= (Bob_QB_0*R)%M,(Bob_QB_1*R)%M
   Bob_RB_0,Bob_RB_1 = (Bob_RB_0*R)%M,(Bob_RB_1*R)%M
   
   mont_one = R%M
   mont_two = (8*R)%M
   A = mont_two
   if A%2==1:
        A = A+M
   else:
        A = A+0
   A = A//2
   if A%2==1:
        A = A+M
   else:
        A = A+0
   A = A//2

   A_0,A_1 = A,0
   R0_X_0,R0_X_1 = Bob_QB_0,Bob_QB_1
   R0_Z_0,R0_Z_1 = mont_one,0
   R2_X_0,R2_X_1 = Bob_RB_0,Bob_RB_1
   R2_Z_0,R2_Z_1 = mont_one,0
   R_X_0,R_X_1 = Bob_PB_0, Bob_PB_1
   R_Z_0,R_Z_1 = mont_one,0

   ind = 0

   (R_X_0,R_X_1,R_Z_0,R_Z_1) = kernelgen(R_X_0,R_X_1,R_Z_0,R_Z_1,R0_X_0,R0_X_1,R0_Z_0,R0_Z_1,R2_X_0,R2_X_1,R2_Z_0,R2_Z_1,key,A_0,A_1,m_prime)

   phiPX_0,phiPX_1 = Alice_PA_0,Alice_PA_1
   phiQX_0,phiQX_1 = Alice_QA_0,Alice_QA_1
   phiDX_0,phiDX_1 = Alice_RA_0,Alice_RA_1

   phiPZ_0,phiPZ_1 = (1*R)%M,0
   phiQZ_0,phiQZ_1=(1*R)%M,0
   phiDZ_0,phiDZ_1=(1*R)%M,0
   A24plus_0,A24plus_1=mont_two,0
   A24minus_0,A24minus_1=(4*R)%M,0
   
   index = 0
   MAX = 239
   npts = 0
   pts_X_0,pts_X_1 = np.zeros(400,dtype=object),np.zeros(400,dtype=object)
   pts_Z_0,pts_Z_1 = np.zeros(400,dtype=object),np.zeros(400,dtype=object)
   pts_index=np.zeros(400)
   ii = 0
   npts_max = 0
   
   pr = MAX
   tpl_count = 0
   iso_count = 0
   inner_loop = 0
   
   for row in range(1,MAX):
       while index<MAX-row:
           inner_loop += 1
           pts_X_0[npts],pts_X_1[npts] = R_X_0,R_X_1
           pts_Z_0[npts],pts_Z_1[npts] = R_Z_0,R_Z_1
           pts_index[npts] = index
           npts = npts+1
           if npts_max<npts:
                npts_max = npts
           m = listx_bob[ii]
           (R_X_0,R_X_1,R_Z_0,R_Z_1) = xTPLe(R_X_0,R_X_1,R_Z_0,R_Z_1,A24plus_0,A24plus_1,A24minus_0,A24minus_1,m)
           ii = ii+1
           index = index+m
       (A24plus_0,A24plus_1,A24minus_0,A24minus_1,K1_0,K1_1,K2_0,K2_1) = get_3_iso(R_X_0,R_X_1,R_Z_0,R_Z_1)
       iso_count = iso_count+npts

       for i in range(npts):
           (pts_X_0[i],pts_X_1[i],pts_Z_0[i],pts_Z_1[i]) = eval_3_iso(pts_X_0[i],pts_X_1[i],pts_Z_0[i],pts_Z_1[i],K1_0,K1_1,K2_0,K2_1)
       (phiPX_0,phiPX_1,phiPZ_0,phiPZ_1) = eval_3_iso(phiPX_0,phiPX_1,phiPZ_0,phiPZ_1,K1_0,K1_1,K2_0,K2_1)
       (phiQX_0,phiQX_1,phiQZ_0,phiQZ_1) = eval_3_iso(phiQX_0,phiQX_1,phiQZ_0,phiQZ_1,K1_0,K1_1,K2_0,K2_1)
       (phiDX_0,phiDX_1,phiDZ_0,phiDZ_1) = eval_3_iso(phiDX_0,phiDX_1,phiDZ_0,phiDZ_1,K1_0,K1_1,K2_0,K2_1)
       R_X_0,R_X_1 = pts_X_0[npts-1],pts_X_1[npts-1]
       R_Z_0,R_Z_1 = pts_Z_0[npts-1],pts_Z_1[npts-1]
       index=pts_index[npts-1]
       npts=npts-1

       if(abort_row!=-1 and row==abort_row):
            break;
 
   (A24plus_0,A24plus_1,A24minus_0,A24minus_1,K1_0,K1_1,K2_0,K2_1) = get_3_iso(R_X_0,R_X_1,R_Z_0,R_Z_1)
   (phiPX_0,phiPX_1,phiPZ_0,phiPZ_1) = eval_3_iso(phiPX_0,phiPX_1,phiPZ_0,phiPZ_1,K1_0,K1_1,K2_0,K2_1)
   (phiQX_0,phiQX_1,phiQZ_0,phiQZ_1) = eval_3_iso(phiQX_0,phiQX_1,phiQZ_0,phiQZ_1,K1_0,K1_1,K2_0,K2_1)
   (phiDX_0,phiDX_1,phiDZ_0,phiDZ_1) = eval_3_iso(phiDX_0,phiDX_1,phiDZ_0,phiDZ_1,K1_0,K1_1,K2_0,K2_1)

   iso_count = iso_count+3
    
   phiPZ_0,phiPZ_1,phiQZ_0,phiQZ_1,phiDZ_0,phiDZ_1 = inv_3_way(phiPZ_0,phiPZ_1,phiQZ_0,phiQZ_1,phiDZ_0,phiDZ_1)
   (phiPX_0,phiPX_1) = mont_mul_karat(phiPZ_0,phiPZ_1,phiPX_0,phiPX_1,m_prime)
   (phiQX_0,phiQX_1) = mont_mul_karat(phiQZ_0,phiQZ_1,phiQX_0,phiQX_1,m_prime)
   (phiDX_0,phiDX_1) = mont_mul_karat(phiDZ_0,phiDZ_1,phiDX_0,phiDX_1,m_prime)
   phiPX_0,phiPX_1 = (mont_mul_fp(phiPX_0,1,m_prime))%M,(mont_mul_fp(phiPX_1,1,m_prime))%M
   phiQX_0,phiQX_1 = (mont_mul_fp(phiQX_0,1,m_prime))%M,(mont_mul_fp(phiQX_1,1,m_prime))%M
   phiDX_0,phiDX_1 = (mont_mul_fp(phiDX_0,1,m_prime))%M,(mont_mul_fp(phiDX_1,1,m_prime))%M
   A24plus_0,A24plus_1 = (mont_mul_fp(A24plus_0,1,m_prime))%M,(mont_mul_fp(A24plus_1,1,m_prime))%M
   A24minus_0,A24minus_1 = (mont_mul_fp(A24minus_0,1,m_prime))%M,(mont_mul_fp(A24minus_1,1,m_prime))%M
   
   return phiPX_0,phiPX_1,phiQX_0,phiQX_1,phiDX_0,phiDX_1,A24plus_0,A24plus_1,A24minus_0,A24minus_1


#--------------------------------------------- Fault Parameters ----------------------------------------------

eps = 8
eB = 239
num_faults = 6
#loop_abort_rows = [(1,None),(2,235),(4,231),(8,223),(16,207),(32,175),(64,112),(127,0),(239,None)]
loop_abort_rows = [(7,None),(8,223),(16,207),(32,175),(64,112),(127,0),(239,None)]
delta_it = 25

#------------------------------------------------- Memory ----------------------------------------------------

kB = None
lst_phi = [None for i in range(eB+1)]
lst_kgp = [None for i in range(eB+1)]
lst_k = [None for i in range(eB+1)]
phi_pub_keys_B = None
lst_fault_output = []

#--------------------------------------------  Helper Functions ----------------------------------------------

def to_mont(X_0,X_1,Z_0,Z_1):

    return (X_0*R)%M,(X_1*R)%M,(Z_0*R)%M,(Z_1*R)%M

def un_mont(X_0,X_1,Z_0,Z_1):
    X_0,X_1 = (mont_mul_fp(X_0,1,m_prime))%M,(mont_mul_fp(X_1,1,m_prime))%M
    Z_0,Z_1 = (mont_mul_fp(Z_0,1,m_prime))%M,(mont_mul_fp(Z_1,1,m_prime))%M
    return X_0,X_1,Z_0,Z_1

def un_proj_mont(X_0,X_1,Z_0,Z_1):
    Z_0,Z_1 = inv_fp2(Z_0,Z_1)
    X_0,X_1 = mont_mul_karat(X_0,X_1,Z_0,Z_1,m_prime)
    return X_0%M,X_1%M

def get_curve(a0,a1,b0,b1):
    n0,n1 = kgp.add_fp2(a0,a1,b0,b1)
    d0,d1 = kgp.sub_fp2(a0,a1,b0,b1)
    d0,d1 = kgp.inverse_fp2(d0,d1)
    n0,n1 = kgp.mul_fp2(n0,n1,d0,d1)
    n0,n1 = kgp.mul_fp2(n0,n1,2,0)
    return n0%p,n1%p

def get_a24plus(a0,a1,b0,b1):
    a0,a1 = get_curve(a0,a1,b0,b1)
    a0,a1 = kgp.add_fp2(a0,a1,2,0)
    d0,d1 = kgp.inverse_fp2(4,0)
    a0,a1 = kgp.mul_fp2(a0,a1,d0,d1)
    a0,a1,_,_ = to_mont(a0,a1,1,0)
    return a0,a1

#--------------------------------------------- Fault Functions -----------------------------------------------

def loop_aborts(key):
    for i in range(num_faults):
        row,_ = loop_abort_rows[i]
        print(f"Fault {i+1} : Aborting after {row} iterations")
        phiPA_0,phiPA_1,phiQA_0,phiQA_1,_,_,A24plus_0,A24plus_1,A24minus_0,A24minus_1 = isogen3(key,Alice_PA_0,Alice_PA_1,Alice_QA_0,Alice_QA_1,Alice_RA_0,Alice_RA_1,Bob_PB_0,Bob_PB_1,Bob_QB_0,Bob_QB_1,Bob_RB_0,Bob_RB_1,row)
        lst_fault_output.append((phiPA_0,phiPA_1,phiQA_0,phiQA_1,A24plus_0,A24plus_1,A24minus_0,A24minus_1))
    print("Loop aborts complete.")

def kernelgen_pts():
    PX_0,PX_1,QX_0,QX_1,A24plus_0,A24plus_1,A24minus_0,A24minus_1 = lst_fault_output[0]
    for i in range(1,num_faults):
        r1,k = loop_abort_rows[i]
        r2,_ = loop_abort_rows[i+1]

        print(f"\nComputation for Fault {i+1}:")
        phiPX_0,phiPX_1,phiQX_0,phiQX_1,_,_,_,_ = lst_fault_output[i]
        num_sol,RX_0,RX_1,_,_ = kgp.kernelgen_pt(PX_0,PX_1,QX_0,QX_1,phiPX_0,phiPX_1,phiQX_0,phiQX_1)
        PZ_0,PZ_1,QZ_0,QZ_1,RZ_0,RZ_1 = 1,0,1,0,1,0
        if(num_sol!=1):
            raise Exception(f"{num_sol} solutions possible for kernel gen point of phi_{r1}.")

        PX_0,PX_1,PZ_0,PZ_1 = to_mont(PX_0,PX_1,PZ_0,PZ_1)
        QX_0,QX_1,QZ_0,QZ_1 = to_mont(QX_0,QX_1,QZ_0,QZ_1)
        RX_0,RX_1,RZ_0,RZ_1 = to_mont(RX_0,RX_1,RZ_0,RZ_1)
        A24plus_0,A24plus_1,_,_ = to_mont(A24plus_0,A24plus_1,1,0)
        A24minus_0,A24minus_1,_,_ = to_mont(A24minus_0,A24minus_1,1,0)
        
        for row in range(r1+1,r2+1):
            print(f"      Evaluating phi_{row}")
            SX_0,SX_1,SZ_0,SZ_1 = xTPLe(RX_0,RX_1,RZ_0,RZ_1,A24plus_0,A24plus_1,A24minus_0,A24minus_1,eB-row-k)
            lst_kgp[row] = (SX_0,SX_1,SZ_0,SZ_1,A24plus_0,A24plus_1,A24minus_0,A24minus_1)
            A24plus_0,A24plus_1,A24minus_0,A24minus_1,K1_0,K1_1,K2_0,K2_1 = get_3_iso(SX_0,SX_1,SZ_0,SZ_1)
            PX_0,PX_1,PZ_0,PZ_1 = eval_3_iso(PX_0,PX_1,PZ_0,PZ_1,K1_0,K1_1,K2_0,K2_1)
            QX_0,QX_1,QZ_0,QZ_1 = eval_3_iso(QX_0,QX_1,QZ_0,QZ_1,K1_0,K1_1,K2_0,K2_1)
            RX_0,RX_1,RZ_0,RZ_1 = eval_3_iso(RX_0,RX_1,RZ_0,RZ_1,K1_0,K1_1,K2_0,K2_1)
            lst_phi[row] = (K1_0,K1_1,K2_0,K2_1)

        PX_0,PX_1 = un_proj_mont(PX_0,PX_1,PZ_0,PZ_1)
        QX_0,QX_1 = un_proj_mont(QX_0,QX_1,QZ_0,QZ_1)
        PX_0,PX_1,_,_ = un_mont(PX_0,PX_1,1,0)
        QX_0,QX_1,_,_ = un_mont(QX_0,QX_1,1,0)
        A24plus_0,A24plus_1,A24minus_0,A24minus_1 = un_mont(A24plus_0,A24plus_1,A24minus_0,A24minus_1)

    print("Computation of isogenies and kernel gen points complete.\n")
    return

#                                 --------- brute force functions ----------

def check_key(k,f):
    thetaP_0,thetaP_1,thetaQ_0,thetaQ_1,_,_,a24plus_0,a24plus_1,a24minus_0,a24minus_1 = isogen3(k,Alice_PA_0,Alice_PA_1,Alice_QA_0,Alice_QA_1,Alice_RA_0,Alice_RA_1,Bob_PB_0,Bob_PB_1,Bob_QB_0,Bob_QB_1,Bob_RB_0,Bob_RB_1,eps-1)
    phiPA_0,phiPA_1,phiQA_0,phiQA_1,A24plus_0,A24plus_1,A24minus_0,A24minus_1 = f
    a0,a1 = get_curve(a24plus_0,a24plus_1,a24minus_0,a24minus_1)
    A0,A1 = get_curve(A24plus_0,A24plus_1,A24minus_0,A24minus_1)
    if((thetaP_0%p,thetaP_1%p,thetaQ_0%p,thetaQ_1%p) == (phiPA_0%p,phiPA_1%p,phiQA_0%p,phiQA_1%p) and (a0,a1)==(A0,A1)):
        return k
    else:
        return None

def next_num():
    i = 0
    while(i<3**eps):
        yield i
        i = i+1

def parallel():
    start = time.time()
    tot = 0
    with Pool(12) as pool:
        for k in pool.imap_unordered(partial(check_key, f=lst_fault_output[0]), next_num(), chunksize=3):
            tot = tot+1
            if k is not None:
                pool.terminate()
                return k
            if tot%delta_it == delta_it-1: 
                now = time.time()
                tt = time.strftime("%H:%M:%S", time.gmtime(now-start))
                print(f"  Timer : {tt} | checked upto k_B' (mod 3**{eps}) = {tot}")
    return None

def serial():
    start = time.time()
    tot = 0
    for k in next_num():
        tot = tot+1
        k = check_key(k,lst_fault_output[0])
        if k is not None:
            return k
        if tot%delta_it == delta_it-1:
            now = time.time()
            tt = time.strftime("%H:%M:%S", time.gmtime(now-start))
            print(f"  Timer : {tt} | checked upto kB' (mod 3**{eps}) = {tot}")
    return None

def brute_force():
    print(f"Starting Brute Force search for kB (mod 3**{eps})")
    #k = serial()
    k = parallel()
    if k is None:
        raise Exception(f"Failed. No value of kB satisfies phi*_{eps}(P_A), phi*_{eps}(P_A) and E_{eps}.")
    print(f"Successful. kB (mod 3**{eps}) = {k}\n")
    global phi_pub_keys_B
    phi_pub_keys_B = isogen3(k,Bob_PB_0,Bob_PB_1,Bob_QB_0,Bob_QB_1,Bob_RB_0,Bob_RB_1,Bob_PB_0,Bob_PB_1,Bob_QB_0,Bob_QB_1,Bob_RB_0,Bob_RB_1,eps-1)
    global kB
    kB = k
    for i in range(1,eps+1):
        lst_k[i] = k%3
        k = k//3
    return

#                                 ------------------------------------------

def compute_kB():
    PX_0,PX_1,QX_0,QX_1,DX_0,DX_1,a24plus_0,a24plus_1,a24minus_0,a24minus_1 = phi_pub_keys_B
    PZ_0,PZ_1,QZ_0,QZ_1,DZ_0,DZ_1 = 1,0,1,0,1,0
    PX_0,PX_1,PZ_0,PZ_1 = to_mont(PX_0,PX_1,PZ_0,PZ_1)
    QX_0,QX_1,QZ_0,QZ_1 = to_mont(QX_0,QX_1,QZ_0,QZ_1)
    DX_0,DX_1,DZ_0,DZ_1 = to_mont(DX_0,DX_1,DZ_0,DZ_1)
    a24plus_0,a24plus_1,a24minus_0,a24minus_1 = to_mont(a24plus_0,a24plus_1,a24minus_0,a24minus_1)

    _,_,_,_,A24plus_0,A24plus_1,A24minus_0,A24minus_1 = lst_kgp[eps+1]
    a0,a1 = get_curve(a24plus_0,a24plus_1,a24minus_0,a24minus_1)
    A0,A1 = get_curve(A24plus_0,A24plus_1,A24minus_0,A24minus_1)
    if((a0,a1)!=(A0,A1)):
        raise Exception(f"The curves of phi_{eps}*(P_B),phi_{eps}*(Q_B) are not the same as that of phi_{eps}*(R_B).")

    print(f"Computing rest of kB:")
    start = time.time()
    global kB 
    md = 3**eps
    for row in range(eps+1,eB+1):
        RX_0,RX_1,RZ_0,RZ_1,A24plus_0,A24plus_1,A24minus_0,A24minus_1 = lst_kgp[row]
        RX_0,RX_1 = un_proj_mont(RX_0,RX_1,RZ_0,RZ_1)
        RZ_0,RZ_1 = R%M,0
        pX_0,pX_1,pZ_0,pZ_1 = xTPLe(PX_0,PX_1,PZ_0,PZ_1,A24plus_0,A24plus_1,A24minus_0,A24minus_1,eB-row)
        qX_0,qX_1,qZ_0,qZ_1 = xTPLe(QX_0,QX_1,QZ_0,QZ_1,A24plus_0,A24plus_1,A24minus_0,A24minus_1,eB-row)
        dX_0,dX_1,dZ_0,dZ_1 = xTPLe(DX_0,DX_1,DZ_0,DZ_1,A24plus_0,A24plus_1,A24minus_0,A24minus_1,eB-row)
        a24plus0,a24plus1 = get_a24plus(A24plus_0,A24plus_1,A24minus_0,A24minus_1)
        found = False
        for i in range(3):
            k = kB + i*md
            rX_0,rX_1,rZ_0,rZ_1 = kernelgen(pX_0,pX_1,pZ_0,pZ_1,qX_0,qX_1,qZ_0,qZ_1,dX_0,dX_1,dZ_0,dZ_1,k,a24plus0,a24plus1,m_prime)
            rX_0,rX_1 = un_proj_mont(rX_0,rX_1,rZ_0,rZ_1)
            rZ_0,rZ_1 = R%M,0
            if((rX_0,rX_1,rZ_0,rZ_1)==(RX_0,RX_1,RZ_0,RZ_1)):
                kB = k
                lst_k[row] = i
                found = True
                break
        if not found:
            raise Exception(f"No possible value of kB (mod 3**{row}) satisfies the equation.")
        now = time.time()
        tt = time.strftime("%H:%M:%S", time.gmtime(now-start))
        print(f"  Timer : {tt} | found kB[{row:003}] = {lst_k[row]} | kB (mod 3**{row}) = {kB}")
        K1_0,K1_1,K2_0,K2_1 = lst_phi[row]
        PX_0,PX_1,PZ_0,PZ_1 = eval_3_iso(PX_0,PX_1,PZ_0,PZ_1,K1_0,K1_1,K2_0,K2_1)
        QX_0,QX_1,QZ_0,QZ_1 = eval_3_iso(QX_0,QX_1,QZ_0,QZ_1,K1_0,K1_1,K2_0,K2_1)
        DX_0,DX_1,DZ_0,DZ_1 = eval_3_iso(DX_0,DX_1,DZ_0,DZ_1,K1_0,K1_1,K2_0,K2_1)
        md = md*3
    print(f"Computation of kB complete.\n")
    return kB

#-------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    key=79043935515841587532978738494542608941554431789956392413907630825384341238190997254716392437274350536343110922300;
    #key=0x3ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff 
    #key=0xee528cd636f4c866460bb278f8a5fd5f52ec9757c25c1ed;
    #key = 1318164695851965694990952793546058101099999169999992999918809180930


    #loop aborts at required positions.
    loop_aborts(key)

    start = time.time()

    #calculating phi_i and corresponding kernel gen pts. for eps<i<=eB.
    kernelgen_pts()

    #brute force to find k[1] to k[eps].
    brute_force()

    #finding k[eps+1] to k[eB]. 
    compute_kB()

    end = time.time()
    tt = time.strftime("%H:%M:%S", time.gmtime(end-start))

    print(f"Fault Attack is complete.\n")
    print(f"FAULT DETAILS:")
    print(f"    1. No. of faults used is {num_faults}")
    print(f"    2. Total time taken for computation : {tt}")
    print(f"    3. Secret key is kB = {kB}")
    if(kB==key):
        print(f"4. Attack was Successful. key and kB match.\n")
    else:
        print(f"4. Attack Failed. key and kB do not match.\n")



   




    
    
