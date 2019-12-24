class Try:

	def __init__(self):
		print("No")

	def __call__(self):
		print("Yes")

t = Try()
t2 = Try()
t2()