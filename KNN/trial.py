h = [1, 1, 3, 3, 4, 1]
b = sorted(h)
print(b)
print(h)

test_str = 'abc'
print(test_str[::-1] == test_str)

res = [test_str[i:j] for i in range(len(test_str)) for j in range(i + 1, len(test_str) + 1)] 

print(res)

b = set()
b.add(4)
b.add(4)
print(len(b))



a = [0, 1, 2, 3, 4]
b = [0, 1, 2, 1, 2]

k = []

for i in range(len(a)):
	k.insert(b[i], a[i])
print(k)

s = 'aabbaa'
N = len(s)
result = 0

for i in range(2*N-1):
	left = i//2
	right = (i+1)//2
	#print("left= ", left,"right = ",right)
	while left >= 0 and right < N and s[left] == s[right]:
		result += 1
		print(s[left:right+1])
		print("***")
		left -= 1
		right += 1
	#print(result)

print(result)

print(s[2:])
"""
print('My decision tree:')
print('branch 0{\n\tdeep: 0\n\tnum of samples for each class: 5 : 9 \n\tsplit by dim 0\n\tbranch 0->0{\n\t\tdeep: 1'
      '\n\t\tnum of samples for each class: 3 : 2 \n\t\tsplit by dim 1\n\t\tbranch 0->0->0{\n\t\t\tdeep: 2\n\t\t\t'
      'num of samples for each class: 3 \n\t\t\tclass:0\n\t\t}\n\t\tbranch 0->0->1{\n\t\t\tdeep: 2\n\t\t\tnum of '
      'samples for each class: 2 \n\t\t\tclass:1\n\t\t}\n\t}\n\tbranch 0->1{\n\t\tdeep: 1\n\t\tnum of samples for '
      'each class: 4 \n\t\tclass:1\n\t}\n\tbranch 0->2{\n\t\tdeep: 1\n\t\tnum of samples for each class: 2 : 3 '
      '\n\t\tsplit by dim 2\n\t\tbranch 0->2->0{\n\t\t\tdeep: 2\n\t\t\tnum of samples for each class: 3 \n\t\t\t'
      'class:1\n\t\t}\n\t\tbranch 0->2->1{\n\t\t\tdeep: 2\n\t\t\tnum of samples for each class: 2 \n\t\t\tclass:0'
      '\n\t\t}\n\t}\n}')

print('My decision tree after pruning:')
print('branch 0{\n\tdeep: 0\n\tnum of samples for each class: 5 : 9 \n\tsplit by dim 0\n\tbranch 0->0{\n\t\tdeep: '
      '1\n\t\tnum of samples for each class: 3 : 2 \n\t\tsplit by dim 1\n\t\tbranch 0->0->0{\n\t\t\tdeep: 2\n\t\t\t'
      'num of samples for each class: 3 \n\t\t\tclass:0\n\t\t}\n\t\tbranch 0->0->1{\n\t\t\tdeep: 2\n\t\t\tnum of '
      'samples for each class: 2 \n\t\t\tclass:1\n\t\t}\n\t}\n\tbranch 0->1{\n\t\tdeep: 1\n\t\tnum of samples for '
      'each class: 4 \n\t\tclass:1\n\t}\n\tbranch 0->2{\n\t\tdeep: 1\n\t\tnum of samples for each class: 2 : 3 '
      '\n\t\tclass:1\n\t}\n}')
"""
