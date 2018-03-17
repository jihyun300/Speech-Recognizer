import math
import copy
import operator
import os
import hmm_header

N_DIMENSION = 39
N_PDF = 10
MUL_BIGRAM=0.01
LOGZERO = -100000000.0

class HMM():
	def __init__(self, numstates):
		self._numstates = numstates
		self._a = {}
		self._mean = {}
		self._variance = {}
		self._gconst = {}
		self._weight = {}

	@property
	def numstates(self):
		return self._numstates
	@property
	def a(self):
		return self._a
	@property
	def mean(self):
		return self._mean
	@property
	def variance(self):
		return self._variance
	@property
	def gconst(self):
		return self._gconst
	@property
	def weight(self):
		return self._weight

	def cal_gconst(self):
		for state,mixnum in self.weight.keys():
			self.gconst[state,mixnum] = 1.0
			for variance in self.variance[state,mixnum]:
				self.gconst[state,mixnum] *= math.sqrt(variance)
			self.gconst[state,mixnum] = 1.0/(math.sqrt(2.0*math.pi)*self.gconst[state,mixnum])

	def b(self,state,x):
		result = {}
		for a,b in self.weight.keys():	#iterate g
			if a != state: continue
			result[b] = 0.0
			for index in range(N_DIMENSION): 	#iterate i
				result[b] += math.pow(x[index] - self.mean[a,b][index],2)/self.variance[a,b][index]
			result[b] = log(self.weight[a,b] * self.gconst[a,b] * exp((-1.0/2.0)*result[b]))
		output = LOGZERO
		for x in result:
			output = logsum(output,result[x])
		return output


def build_unit_hmm(hmmlist):
	for name in range(len(hmm_header.phones)):
		sound=hmm_header.phones[name][0]
		numstates=len(hmm_header.phones[name][1])
	
		hmmlist[sound]=HMM(numstates)

		for i in range(numstates):
			hmmlist[sound].a[i]=hmm_header.phones[name][1][i]

		for state in range(len(hmm_header.phones[name][2])):
			for pdf in range(N_PDF):
				hmmlist[sound].weight[state+1,pdf+1]=hmm_header.phones[name][2][state][pdf][0]
				hmmlist[sound].mean[state+1,pdf+1]=hmm_header.phones[name][2][state][pdf][1]
				hmmlist[sound].variance[state+1,pdf+1]=hmm_header.phones[name][2][state][pdf][2]



def viterbi(hmm,x):
	delta = {}
	psi = {}
	for j in range(hmm.numstates):
		delta[1,j] = logproduct(log(hmm.a[0][j]),hmm.b(j,x[1]))
		psi[1,j] = 0
	for t in range(2,len(x)+1):
		for j in range(hmm.numstates):
			delta[t,j] = LOGZERO
			psi[t,j]=0
			for i in range(hmm.numstates):
				if delta[t,j] < logproduct(delta[t-1,i], log(hmm.a[i][j])):
					delta[t,j] = logproduct(delta[t-1,i] , log(hmm.a[i][j]))
					psi[t,j] = i
			if j == hmm.numstates -1:
				delta[t,j] = LOGZERO
			else:
				delta[t,j] = logproduct(delta[t,j], hmm.b(j,x[t]))
	

	
	delta[len(x),hmm.numstates-1] = LOGZERO	
	for j in range(hmm.numstates):
		if delta[len(x),hmm.numstates-1] < logproduct(delta[len(x),j],log(hmm.a[j][hmm.numstates-1])):
				delta[len(x),hmm.numstates-1] = logproduct(delta[len(x),j],log(hmm.a[j][hmm.numstates-1]))
				psi[len(x),hmm.numstates-1] = j
	


	q = [ 0 for i in range(len(x)) ]
	q[len(x) - 1] = psi[len(x),hmm.numstates-1]
	for t in range(len(x)-2,-1,-1):
		q[t] = psi[t+1,q[t+1]]
	return q	


def append_hmm(front, end):
	#make new_hmm whose number of states is front+end states -2 and copy front
	
	if front.numstates == 0:
		return end;

	numstates = front.numstates + end.numstates -2
	new_hmm = HMM(numstates)
	new_hmm.mean = copy.deepcopy(front.mean)
	new_hmm.variance = copy.deepcopy(front.variance)
	new_hmm.weight = copy.deepcopy(front.weight)
	
	#append end hmm to new_hmm upadting indexes of states
	for state,mixnum in end.mean:
		new_hmm.mean[state+front.numstates-2,mixnum] = copy.deepcopy(end.mean[state,mixnum])
		new_hmm.variance[state+front.numstates-2,mixnum] = copy.deepcopy(end.variance[state,mixnum])
		new_hmm.weight[state+front.numstates-2,mixnum] = copy.deepcopy(end.weight[state,mixnum])
	
	#upadte matrix a
	for row in range(numstates):
		new_hmm.a[row] = [0 for x in range(numstates)]
		for col in range(numstates):
			if row == front.numstates - 2 and col == front.numstates -1:	
				p = front.a[row][col]
				end.a[0] = [x*p for x in end.a[0]]
			if row >= front.numstates -2 and col >= front.numstates -1 and row <= numstates - 1 and col <= numstates -1:
				new_hmm.a[row][col] = end.a[row - front.numstates+2][col-front.numstates+2]
			elif row < front.numstates -1 and col < front.numstates -1:
				new_hmm.a[row][col] = front.a[row][col]
	
	#cal gconst
	new_hmm.cal_gconst()
	
	return new_hmm

def exp(x):
	if x == LOGZERO:
		return 0
	else:
		return math.exp(x)

def log(x):
	if x == 0:
		return LOGZERO
	elif x > 0:
		return math.log(x)

def logproduct(x,y):
	if x == LOGZERO or y == LOGZERO:
		return LOGZERO
	else:
		return x+y

def logsum(x,y):
	if x == LOGZERO or y == LOGZERO :
		if x == LOGZERO:
			return y
		else:
			return x
	else:
		if x > y :
			return x + log(1+math.exp(y-x))
		else:
			return y + log(1+math.exp(x-y))


#Main
dic = open("dictionary.txt","r")
bi = open("bigram.txt","r")
total_states = 0
hmmlist = {}
words = {}
unigram = {}
bigram = {}


#set unigram from bigram
unigram["eight"]=0.012084
unigram["five"]=0.011881
unigram["four"]=0.009139
unigram["nine"]=0.011474
unigram["oh"]=0.012591
unigram["one"]=0.010967
unigram["seven"]=0.010967
unigram["six"]=0.011779
unigram["three"]=0.010865
unigram["two"]=0.013201
unigram["zero"]=0.010053



#get bigram
print "1. reading bigram"
for line in bi:
	temp = line.split()
	bigram[temp[0],temp[1]] = float(temp[2])
bi.close()

print "2. reading dictionary"

for line in dic:
	temp = line.split()
	if temp[0] == "zero" and temp[2] == "iy":
		words["zero2"] = temp[1:]
	else:
		words[temp[0]] = temp[1:]

dic.close()

#read hmm_header and build unit HMMs
print "3. building Unit HMM"
build_unit_hmm(hmmlist)

#calculate gconst value
for hmm in hmmlist.keys():	
	hmmlist[hmm].cal_gconst()



print "4. building Word HMM"
word_hmm = {}
for word in words:
	word_hmm[word] = copy.deepcopy(hmmlist[words[word][0]])	
	for index in range(len(words[word])-1):
		word_hmm[word] = append_hmm( word_hmm[word], copy.deepcopy(hmmlist[words[word][index+1]]) )
		

print "5. building Final HMM"
start_index = 1
hmm_index = {}	
end_index = 0

hmm = HMM(0)
for word in words:
	hmm = append_hmm(hmm,word_hmm[word])
	end_index = hmm.numstates - 1
	hmm_index[word] = (start_index,end_index)
	start_index = end_index + 1

for word_row in hmm_index:
	(a,b) = hmm_index[word_row]
	if word_row == "zero2":
		hmm.a[0][a] = unigram["zero"]
	else:
		hmm.a[0][a] = unigram[word_row]

	temp_index = 0
	for word_col in hmm_index:
		(c,d) = hmm_index[word_col]
		temp_index = word_hmm[word_row].numstates-1	
		hmm.a[b-1][c]=0.0
		hmm.a[b][c]=0.0
		if word_row == "zero2":
			if word_col == "zero2":
				hmm.a[b-1][c+3] = bigram["zero","zero"] * word_hmm[word_row].a[temp_index-2][temp_index] * MUL_BIGRAM
				hmm.a[b][c+3] = bigram["zero","zero"] * word_hmm[word_row].a[temp_index-1][temp_index] * MUL_BIGRAM
			else:
				hmm.a[b-1][c+3] = bigram["zero",word_col] * word_hmm[word_row].a[temp_index-2][temp_index] * MUL_BIGRAM
				hmm.a[b][c+3] = bigram["zero",word_col] * word_hmm[word_row].a[temp_index-1][temp_index] * MUL_BIGRAM
		else:
			if word_col == "zero2":
				hmm.a[b-1][c+3] = bigram[word_row,"zero"] * word_hmm[word_row].a[temp_index-2][temp_index] * MUL_BIGRAM
				hmm.a[b][c+3] = bigram[word_row,"zero"] * word_hmm[word_row].a[temp_index-1][temp_index] * MUL_BIGRAM
			else:
				hmm.a[b-1][c+3] = bigram[word_row,word_col] * word_hmm[word_row].a[temp_index-2][temp_index] * MUL_BIGRAM
				hmm.a[b][c+3] = bigram[word_row,word_col] * word_hmm[word_row].a[temp_index-1][temp_index] * MUL_BIGRAM

for word_row in hmm_index:
	(a,b) = hmm_index[word_row]
	hmm.a[b-1][hmm.numstates - 1] = word_hmm[word_row].a[word_hmm[word_row].numstates-2][word_hmm[word_row].numstates-1]
	hmm.a[b][hmm.numstates - 1] = word_hmm[word_row].a[word_hmm[word_row].numstates-1][word_hmm[word_row].numstates-1]

"""
for word in hmm_index:
	print word, 
	print "\t",
	print hmm_index[word]
"""
# calculate veterbi and get answers
print "----------------Start calculate and get answers-----------------------"

answer_word = []
result = open("recognized.txt","w")
result.write("#!MLF!#\n")
cnt=0
for root, dirs, files in os.walk("tst", topdown = False):
	root=root.replace("\\","/")
	for name in files:
		result.write("\"" + root+"/"+name[0:-3] + "rec\"\n")
		test_file = open(os.path.join(root,name))
		length = int(test_file.readline().split()[0]);
		x = {}
		for i in range(1,length+1):
			x[i] = map(float,test_file.readline().split())
		cnt+=1
		print "%d. %s/%s" %(cnt,root,name)
		answer = viterbi(hmm,x) # state list

		#convert state to word
		answer_word=[]
		cur_word = ""
		for i in range(len(answer)-1):
			for word in hmm_index:
				(a,b) = hmm_index[word]
				if answer[i] <= b and answer[i] >= a+3 and cur_word != word:
					cur_word = word
					print word
					if cur_word == "zero2":
						answer_word.append("zero")	
					else:
						answer_word.append(cur_word) 
		# write words to file
		for i in range(len(answer_word)):
			result.write(answer_word[i])
			result.write("\n")
		result.write(".\n")











