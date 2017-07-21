import json

class Bot(object):
	def __init__(self):

		self.count = 0 #Number of game iterations run
		file = open ("Run.txt", "r")
		self.count = int(file.readline())
		file.close()

		self.dumpInterval = 20 # Number of game iterations until json dump

		# Rewards dictionary 0 = alive 1 = dead
		self.rewards = {0:1, 1:-1000}

		# Load Q values
		self.load_qval()

		self.learning_rate = 0.7
		self.action = 0
		self.state = "420_240_0"
		self.discount = 1.0
		self.moves = []

	def load_qval(self):
		self.qval = {}
		try:
			file = open("qvalues.json", "r")
		except:
			print("Error reading q values")
		self.qval = json.load(file)
		file.close()

	def act(self, x, y, vel):
		state = self.getMapState(x, y, vel)

		self.moves.append([self.state, self.action, state])

		self.state = state

		if self.qval[state][0] >= self.qval[state][1]:
			self.action = 0
			return 0
		else:
			self.action = 1
			return 1

	def getMapState(self, x, y, vel):
		if x < 140:
			x = int(x) - (int(x) % 10)
		else:
			x = int(x) - (int(x) % 70)

		if y < 180:
			y = int(y) - (int(y) % 10)
		else:
			y = int(y) - (int(y) % 60)

		return str(int(x)) + "_" + str(int(y)) + "_" + str(vel) 

	def update(self):
		history = list(reversed(self.moves))

		# Check if died from top pipe
		if int(history[0][2].split("_")[1]) > 120:
			top_die = True
		else:
			top_die = False

		# Update Q array using Q Learning rule
		t = 1
		for exp in history:
			state = exp[0]
			act = exp[1]
			res_state = exp[2]
			if t == 1 or t == 2:
				self.qval[state][act] = (1- self.learning_rate) * (self.qval[state][act]) + (self.learning_rate) * ( self.rewards[1] + (self.discount)*max(self.qval[res_state]) )
			elif act and top_die:
				top_die = False
				self.qval[state][act] = (1- self.learning_rate) * (self.qval[state][act]) + (self.learning_rate) * ( self.rewards[1] + (self.discount)*max(self.qval[res_state]) )
			else:
				self.qval[state][act] = (1- self.learning_rate) * (self.qval[state][act]) + (self.learning_rate) * ( self.rewards[0] + (self.discount)*max(self.qval[res_state]) )
			t += 1

		self.count += 1 # increment game counter
		# Change counter
		file = open ("Run.txt", "w")
		file.write(str(self.count))
		file.close()
		self.dump() 
		self.moves = [] # Clear moves

	def progress(self, score):
		file = open ("progress.txt", "a")
		file.write("Iteration: " + str(self.count) + " Score: " + str(score) + "\n")
		file.close()

	def dump(self):
		if self.count % self.dumpInterval == 0:
			file = open("qvalues.json", "w")
			json.dump(self.qval, file)
			file.close()
			print("Q-Values updates")

	def getLastState(self):
		return self.state

