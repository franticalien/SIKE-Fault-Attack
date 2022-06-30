from random import randint

#---------------------------------------- prime number 'p' -------------------------------------

p=(2**372)*(3**239)-1 # choose a prime number p of form 4(mod 3).
#p = 2**31 - 1
#p = 23
#p = 7

#----------------------------------------- Fp^2 functions --------------------------------------

def inverse_fp(x):
	prime = p
	res = 1
	a,b = x,p-2
	while(b>0):
		if(b%2==1):
			res = (res*a)%p
		a = (a*a)%p
		b = b//2
	if (x*res)%p != 1:
		return None 
	return res

def mul_fp2(A0,A1,B0,B1):
	#Input: A,B
	#Output: A*B
	A0,A1,B0,B1 = A0%p,A1%p,B0%p,B1%p
	x = ( (A0*B0)%p - (A1*B1)%p )%p
	y = ( (A0*B1)%p + (A1*B0)%p )%p
	return x,y

def add_fp2(A0,A1,B0,B1):
	#Input: A,B
	#Output: A+B
	A0,A1,B0,B1 = A0%p,A1%p,B0%p,B1%p
	x = (A0+B0)%p
	y = (A1+B1)%p  
	return x,y

def sub_fp2(A0,A1,B0,B1):
	#Input: A,B
	#Output: A-B
	A0,A1,B0,B1 = A0%p,A1%p,B0%p,B1%p
	x = (A0-B0)%p
	y = (A1-B1)%p  
	return x,y

def sq_fp2(A0,A1):
	#Input: A
	#Output: A**2
	return mul_fp2(A0,A1,A0,A1)

def modpow_fp2(a0,a1,b):
	#Input: A,b
	#Output: A**b
    a0,a1 = a0%p,a1%p
    x0,x1 = 1,0
    while(b>0):
        if(b%2==1):
            x0,x1 = mul_fp2(x0,x1,a0,a1)
        a0,a1 = sq_fp2(a0,a1)
        b = b//2
    return x0%p,x1%p

def inverse_fp2(a0,a1):
	#Input: A
	#Output: A**(-1)
	a0,a1 = a0%p,a1%p
	if(a0%p==0 and a1%p==0):
		print("Inverse DNE.")
		return None
	x,_ = mul_fp2(a0,a1,a0,-a1)
	x = inverse_fp(x)
	x0,x1 = (a0*x)%p, ((p-a1)*x)%p
	return x0,x1

def sqrt_fp2(a0,a1):
	#Input: A
	#Output: sqrt(A) - only one possibility. The other can be calculated as [p-sqrt(A)].
	#Note: Only works if p is of form 4(mod 3).e
    a0,a1 = a0%p,a1%p
    ta_0, ta_1 = modpow_fp2(a0,a1,(p-1)//2)
    tb_0, tb_1 = modpow_fp2(a0,a1,(p+1)//4)

    if((ta_0+1)%p == 0 and ta_1%p == 0):
        tb_0, tb_1 = mul_fp2(tb_0,tb_1,0,1)
        chk0,chk1 = sq_fp2(tb_0,tb_1)
        if(chk0==a0 and chk1==a1):
        	return tb_0%p,tb_1%p
        else:
        	print("Not a QR.")
        	return None, None

    ta_0, ta_1 = add_fp2(ta_0,ta_1,1,0)
    ta_0, ta_1 = modpow_fp2(ta_0,ta_1,(p-1)//2)
    ta_0, ta_1 = mul_fp2(ta_0,ta_1,tb_0,tb_1)
    chk0,chk1 = sq_fp2(ta_0,ta_1)
    if(chk0==a0 and chk1==a1):
        return ta_0%p,ta_1%p
    else:
    	print("Not a QR.")
    	return None, None

def quad_fp2(A0,A1,B0,B1,C0,C1):
	#Input: A,B,C
	#Output: X1,X2 - where these are solutions to A*(X**2) + B*(X) + C = 0.
    A0,A1,B0,B1,C0,C1 = A0%p,A1%p,B0%p,B1%p,C0%p,C1%p
    if(A0%p==0 and A1%p==0):
        if(B0%p==0 and B1%p==0):
            if(C0%p==0 and C1%p==0):
            	print("Cannot solve Identity Equation.")
            	return "True", "True", "True", "True"
            else:
            	print("No solution exists.")
            	return "False", "False", "False", "False"
        else:
            C0,C1 = sub_fp2(0,0,C0,C1)
            B0,B1 = inverse_fp2(B0,B1)
            C0,C1 = mul_fp2(C0,C1,B0,B1)
            return C0,C1,C0,C1

    d0,d1 = mul_fp2(A0,A1,2,0)
    d0,d1 = inverse_fp2(d0,d1)

    na_0,na_1 = sub_fp2(0,0,B0,B1)
    nb_0,nb_1 = sq_fp2(B0,B1)
    nc_0,nc_1 = mul_fp2(A0,A1,4,0)
    nc_0,nc_1 = mul_fp2(nc_0,nc_1,C0,C1)
    nb_0,nb_1 = sub_fp2(nb_0,nb_1,nc_0,nc_1)
    nb_0,nb_1 = sqrt_fp2(nb_0,nb_1)

    x0,x1 = add_fp2(na_0,na_1,nb_0,nb_1)
    x0,x1 = mul_fp2(x0,x1,d0,d1)
    y0,y1 = sub_fp2(na_0,na_1,nb_0,nb_1)
    y0,y1 = mul_fp2(y0,y1,d0,d1)

    return x0,x1,y0,y1

def common_quad_fp2(A0,A1,B0,B1,C0,C1,P0,P1,Q0,Q1,R0,R1):
	#Input: A,B,C,P,Q,R
	#Output: X1,X2 - where these are common solutions to A*(X**2) + B*(X) + C = 0 and P*(X**2) + Q*(X) + R = 0.
	x0,x1,y0,y1 = quad_fp2(A0,A1,B0,B1,C0,C1)
	w0,w1,z0,z1 = quad_fp2(P0,P1,Q0,Q1,R0,R1)
	if(x0=="True"):
		return w0,w1,z0,z1
	if(w0=="True"):
		return x0,x1,y0,y1
	if(x0=="False" or w0=="False"):
		return "False", "False", "False", "False"
	s1,s2 = set(),set()
	s1.add((x0,x1))
	s1.add((y0,y1))
	s2.add((w0,w1))
	s2.add((z0,z1))
	s = s1.intersection(s2)
	if(len(s)==0):
		return "False", "False", "False", "False"
	elif(len(s)==1):
		x0,x1 = s.pop()
		return x0,x1,x0,x1
	elif(len(s)==2):
		x0,x1 = s.pop()
		y0,y1 = s.pop()
		return x0,x1,y0,y1
	else:
		print("Undefined Behaviour.")
		return None, None, None, None

#-------------------------------------- specific functions -------------------------------------

def calc_coeff_from_roots(g1,g2,a1,a2,b1,b2):
	#Input: Gamma,Alpha,Beta
	#Output: A,B,C - such that A*(X**2) + B*(X) + C = Gamma*(X-Alpha)*(X-Beta).
	A0,A1 = g1,g2
	B0,B1 = add_fp2(a1,a2,b1,b2)
	B0,B1 = mul_fp2(B0,B1,-g1,-g2)
	C0,C1 = mul_fp2(a1,a2,b1,b2)
	C0,C1 = mul_fp2(C0,C1,g1,g2)
	return A0,A1,B0,B1,C0,C1

def calc_coeff_kernelgen(a,b,x,y):
	#Input: P,phi(P) - where kernel generator pt. of phi is X(say).
	#Output: A,B,C - such that the equation in X, between P and phi(P) reduces to A*(X**2) + B*(X) + C = 0.
	A0,A1 = modpow_fp2(a,b,3)
	A0,A1 = sub_fp2(A0,A1,x,y)
	B0,B1 = sub_fp2(x,y,a,b)
	B0,B1 = mul_fp2(B0,B1,a,b)
	B0,B1 = mul_fp2(B0,B1,2,0)
	C0,C1 = mul_fp2(a,b,x,y)
	C0,C1 = sub_fp2(1,0,C0,C1)
	C0,C1 = mul_fp2(C0,C1,a,b)
	return A0,A1,B0,B1,C0,C1

def kernelgen_pt(p0,p1,q0,q1,phi_p0,phi_p1,phi_q0,phi_q1):
	#Input: P,Q,phi(P),phi(Q) - where kernel generator pt. of phi is R(say).
	#Output: n,X,Y - where n is number of possible solutions to R and X,Y are the possible solutions.
	A0,A1,B0,B1,C0,C1 = calc_coeff_kernelgen(p0,p1,phi_p0,phi_p1)
	P0,P1,Q0,Q1,R0,R1 = calc_coeff_kernelgen(q0,q1,phi_q0,phi_q1)
	x0,x1,y0,y1 = common_quad_fp2(A0,A1,B0,B1,C0,C1,P0,P1,Q0,Q1,R0,R1)
	if(x0=="True"):
		print("All points are possible kernelgen pts.")
		return p**2, "True", "True", "True", "True"
	elif(x0=="False"):
		print("No possible kernelgen pts.")
		return 0, "False", "False", "False", "False"
	elif(x0==y0 and x1==y1):
		return 1,x0,x1,x0,x1
	else:
		return 2,x0,x1,y0,y1

def image_from_kernelgen(p0,p1,r0,r1):
	#Input: P,R
	#Output: phi(P) - where kernel generator pt. of phi is R.
	x0,x1 = mul_fp2(p0,p1,r0,r1)
	x0,x1 = sub_fp2(x0,x1,1,0)
	x0,x1 = sq_fp2(x0,x1)
	x0,x1 = mul_fp2(p0,p1,x0,x1)
	y0,y1 = sub_fp2(p0,p1,r0,r1)
	y0,y1 = sq_fp2(y0,y1)
	y0,y1 = inverse_fp2(y0,y1)
	x0,x1 = mul_fp2(x0,x1,y0,y1)
	return x0,x1

#------------------------------------------- main() ----------------------------------------------

if __name__ == '__main__':

	# DESCRIPTION:
	# 1. I choose N different tuples of P,Q,R.
	# 2. For each tuple, I calculate phi(P) and phi(Q) with kernel gen. pt. of phi as R.
	# 3. Then I use the above functions to calculate R from P,Q,phi(P),phi(Q).
	# 4. Then I check whether this calculated value is equal to the actual value or not.
	# 5. I also keep track of the number of possible solutions to R that is calculated and hence, after
	#    N tries,  an approximate probability of getting 2 possible solutions to R is calculated.
	#    Observation - This probability reduces as 'p' increases.

	N = 10 # N = number of tries. My system takes approx N*0.075 secs to run for p=(2**372)*(3**239)-1.

	tot = 0 # counter for number of valid tries.
	roots = [0,0,0,0] # roots[i] = counter for number of tries for which the no. of possible values of kernel gen. point
    	              # is 'i' (if 0<=i<=2). roots[3] is counter for tries when all pts. could've been the kernel gen pt.

	for i in range(N):
		p0,p1,q0,q1,r0,r1 = (randint(0,p-1) for j in range(6))

    	#----- rejecting (invalid) cases which (I assume) are not possible in SIKE algo. -------
		if ((p0,p1)==(0,0) or (p0,p1)==(1,0) or (p0,p1)==(p-1,0)):
			continue;
		if ((q0,q1)==(0,0) or (q0,q1)==(1,0) or (q0,q1)==(p-1,0)):
			continue;
		if ((p0==q0 and p1==q1) or (p0==r0 and p1==r1) or (q0==r0 and q1==r1)):
			continue;
		#---------------------------------------------------------------------------------------
	
		tot += 1 # updating number of valid tries.

		phi_p0,phi_p1 = image_from_kernelgen(p0,p1,r0,r1)
		phi_q0,phi_q1 = image_from_kernelgen(q0,q1,r0,r1)
		num,x0,x1,y0,y1 = kernelgen_pt(p0,p1,q0,q1,phi_p0,phi_p1,phi_q0,phi_q1)

		if (num>=0 and num<=2): # updating roots[]
			roots[num] += 1
		elif num==p**2:
			roots[3] += 1

		if not ((x0==r0 and x1==r1) or (y0==r0 and y1==r1)): # checking whether the calculated solution is correct.
			print((r0,r1,p0,p1,q0,q1,phi_p0,phi_p1,phi_q0,phi_q1))

	#--------------------------------------- logistics -----------------------------------------
	print(tot)
	assert(tot==sum(roots))
	print(roots)
	print(f"probability of 2 possible values of R = {roots[2]/sum(roots)}")


