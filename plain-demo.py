names = [ 'Elaine', 'Chris', 'David', 'Peter', 'PyCon Canada' ]

for name in names:
    if not (name < 'Peter'):
        break

    print(name)

'''
for name in names while name < 'Peter':
    print(name)
'''
